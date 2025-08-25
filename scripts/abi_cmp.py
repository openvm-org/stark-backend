"""
Script for comparing ABI compatibility between C++ headers and compiled binaries.

This script analyzes C++ header files and compiled ELF binaries to verify that the
function declarations in headers match the actual compiled functions. It helps catch
ABI mismatches and missing/extra functions by:

1. Parsing C++ headers to extract declared function names
2. Reading DWARF debug info from ELF binaries to get compiled function names
3. Comparing the two sets to identify discrepancies
"""

import sys
import os
from os import path
import logging

class CustomFormatter(logging.Formatter):
    gray = "\x1b[38;1m"
    green = "\x1b[32;1m"
    yellow = "\x1b[33;1m"
    red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(levelname)s]{}: %(message)s".format(reset)

    FORMATS = {
        logging.DEBUG: gray + format,
        logging.INFO: green + format,
        logging.WARNING: yellow + format,
        logging.ERROR: red + format,
        logging.CRITICAL: red + format
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logger = logging.getLogger(__name__)
logger.setLevel(LOGLEVEL)
ch = logging.StreamHandler()
ch.setLevel(LOGLEVEL)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# dwarf_compare.py (robust DWARF helpers)

from elftools.elf.elffile import ELFFile

def header_function_names(header_path):
    import re, pathlib
    pat = re.compile(r'^[^#\n{};]*\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*;', re.M)
    txt = pathlib.Path(header_path).read_text()
    return set(pat.findall(txt))

def best_name(sp_die):
    for k in ('DW_AT_MIPS_linkage_name','DW_AT_linkage_name','DW_AT_name'):
        a = sp_die.attributes.get(k)
        if a: return a.value.decode('utf-8','ignore')
    return None

def resolve_final_type(die):
    seen=set()
    while die and 'DW_AT_type' in die.attributes and die.offset not in seen:
        seen.add(die.offset)
        die = die.get_DIE_from_attribute('DW_AT_type')
    return die

def width_of_type(die, cu_addr_size):
    if die is None: return 'void'
    tag = die.tag
    if tag in ('DW_TAG_pointer_type','DW_TAG_reference_type','DW_TAG_rvalue_reference_type'):
        return f'ptr{cu_addr_size*8}'
    if tag == 'DW_TAG_base_type':
        size_attr = die.attributes.get('DW_AT_byte_size')
        if not size_attr: return 'agg?'
        bits = int(size_attr.value)*8
        enc = die.attributes.get('DW_AT_encoding')
        if enc and enc.value in (4,5):  # float/complex
            return f'f{bits}'
        return f'i{bits}'
    size_attr = die.attributes.get('DW_AT_byte_size')
    return f'agg{int(size_attr.value)*8}' if size_attr else 'agg?'

# --- helpers: signedness-aware, no pointer chasing ---

ENC_SIGNED       = {5, 6}   # DW_ATE_signed, DW_ATE_signed_char
ENC_UNSIGNED     = {7, 8}   # DW_ATE_unsigned, DW_ATE_unsigned_char
ENC_FLOAT        = {4, 3, 9}  # float, complex_float, imaginary_float (treat as float)

QUAL_TAGS = {
    'DW_TAG_const_type', 'DW_TAG_volatile_type', 'DW_TAG_restrict_type',
    'DW_TAG_atomic_type', 'DW_TAG_typedef'
}
POINTERY_TAGS = {
    'DW_TAG_pointer_type', 'DW_TAG_reference_type', 'DW_TAG_rvalue_reference_type',
    'DW_TAG_subroutine_type'
}
AGG_TAGS = {
    'DW_TAG_structure_type', 'DW_TAG_class_type', 'DW_TAG_union_type',
    'DW_TAG_array_type', 'DW_TAG_enumeration_type'
}

def peel_qualifiers(die):
    """Follow only qualifiers/typedefs; stop before pointers/refs/aggregates/base."""
    seen = set()
    while die and die.tag in QUAL_TAGS and 'DW_AT_type' in die.attributes and die.offset not in seen:
        seen.add(die.offset)
        die = die.get_DIE_from_attribute('DW_AT_type')
    return die

def type_name(die):
    if not die: return None
    a = die.attributes.get('DW_AT_name')
    return a.value.decode('utf-8','ignore') if a else None

def classify_type(die, cu_addr_size, with_names=False):
    """
    Returns one of:
      - 'ptrNN' for pointers/refs/function-pointers (NN = address bits),
      - 'uNN'/'iNN' for integers, 'fNN' for floats,
      - 'aggNN' (or 'aggNN:Type') for by-value aggregates,
      - 'void' or 'agg?' as last resorts.
    """
    if die is None:
        return 'void'
    die = peel_qualifiers(die)

    if die.tag in POINTERY_TAGS:
        # treat pointers/references and function pointers uniformly
        return f'ptr{cu_addr_size*8}'

    if die.tag == 'DW_TAG_base_type':
        size_attr = die.attributes.get('DW_AT_byte_size')
        enc_attr  = die.attributes.get('DW_AT_encoding')
        if not size_attr:
            return 'agg?'
        bits = int(size_attr.value) * 8
        enc  = int(enc_attr.value) if enc_attr else None
        if enc in ENC_FLOAT:
            return f'f{bits}'
        if enc in ENC_UNSIGNED:
            return f'u{bits}'
        # default: signed (covers booleans too, but width stays correct)
        return f'i{bits}'

    if die.tag in AGG_TAGS:
        size_attr = die.attributes.get('DW_AT_byte_size')
        if size_attr:
            bits = int(size_attr.value) * 8
            if with_names:
                nm = type_name(die)
                return f'agg{bits}' + (f':{nm}' if nm else '')
            return f'agg{bits}'
        return 'agg?' + (f':{type_name(die)}' if with_names and type_name(die) else '')

    # Unknown tag; try size, else give up politely
    size_attr = die.attributes.get('DW_AT_byte_size')
    if size_attr:
        return f'agg{int(size_attr.value)*8}'
    return 'agg?'

def params_from_subprogram(sp_die, cu_addr_size, with_names=False):
    # Prefer explicit child parameters; else use subroutine_type children
    params = [c for c in sp_die.iter_children() if c.tag=='DW_TAG_formal_parameter']
    if not params:
        t_attr = sp_die.attributes.get('DW_AT_type')
        if t_attr:
            tdie = sp_die.get_DIE_from_attribute('DW_AT_type')
            if tdie and tdie.tag=='DW_TAG_subroutine_type':
                params = [c for c in tdie.iter_children() if c.tag=='DW_TAG_formal_parameter']
    out = []
    for p in params:
        tattr = p.attributes.get('DW_AT_type')
        die = p.get_DIE_from_attribute('DW_AT_type') if tattr else None
        out.append(classify_type(die, cu_addr_size, with_names=with_names))
    return out

def load_restricted(obj_path, wanted_names):
    with open(obj_path, 'rb') as f:
        elf = ELFFile(f)
        if not elf.has_dwarf_info():
            return {}
        dwarf = elf.get_dwarf_info()
        sigs = {}
        for cu in dwarf.iter_CUs():
            cu_addr_size = cu['address_size']
            for die in cu.iter_DIEs():
                if die.tag != 'DW_TAG_subprogram': continue
                name = best_name(die)
                if not name or name not in wanted_names: 
                    continue
                params = params_from_subprogram(die, cu_addr_size)
                if name in sigs and sigs[name] != params:
                    logger.error(f"{name} has two different signatures in Rust: {params} and {sigs[name]}")
                    sys.exit(1)
                sigs[name] = params
        return sigs


def is_from_my_header(sp_die, files):
    f = sp_die.attributes.get('DW_AT_decl_file')
    if not f: return False
    path = files.get(int(f.value), "")
    return path.endswith("rust_abi.hpp") or path.endswith("rust_stub.cpp")


def load(path):
    logger.debug(f"Loading CUDA file: {path}")
    with open(path, 'rb') as f:
        elf = ELFFile(f)
        if not elf.has_dwarf_info():
            return {}
        dwarf = elf.get_dwarf_info()
        sigs = {}
        for cu in dwarf.iter_CUs():
            lp = dwarf.line_program_for_CU(cu)
            files = {i+1: fe.name.decode('utf-8','ignore') for i, fe in enumerate(lp['file_entry'])}
            cu_addr_size = cu['address_size']
            for die in cu.iter_DIEs():
                if die.tag != 'DW_TAG_subprogram':
                    continue
                name = best_name(die)
                if not name:
                    continue
                sigs[name] = params_from_subprogram(die, cu_addr_size)
        return sigs

def compare(name, r, c):
    def len_before_number(x):
        i = len(x)
        while i > 0 and x[i - 1].isdigit():
            i -= 1
        return i
    def last_number(x):
        i = len_before_number(x)
        assert i < len(x)
        return int(x[i:])
    if len(r) != len(c):
        return [], "The number of arguments doesn't match"
    warns = []
    errs = []
    for i in range(len(r)):
        if r[i] == c[i]:
            continue
        if r[i].endswith("?") or c[i].endswith("?"):
            continue # for now
        r_pr = len_before_number(r[i])
        c_pr = len_before_number(c[i])
        r_sz = last_number(r[i])
        c_sz = last_number(c[i])
        if r_sz < c_sz:
            errs.append(f"Argument #{i+1} (1-indexed): Rust provides fewer bits than CUDA reads ({r_sz} < {c_sz})")
        elif r_sz > c_sz:
            warns.append(f"{name}: Argument #{i+1} (1-indexed): Rust provides more bits than CUDA reads ({r_sz} > {c_sz})")
        if r[i][:r_pr] != c[i][:c_pr]:
            warns.append(f"Argument #{i+1} (1-indexed): different base types (Rust {r[i][:r_pr]}, CUDA {c[i][:c_pr]})")
    return warns, errs

wanted = header_function_names('rust_abi.hpp')
rust = load_restricted(sys.argv[1], wanted)
cuda = {}
for filename in os.listdir(sys.argv[2]):
    if path.splitext(filename)[1] == ".o":
        cuda.update(load(path.join(sys.argv[2], filename)))
ok = True
for name in sorted(set(rust) | set(cuda)):
    r, c = rust.get(name), cuda.get(name)
    if r and c:
        warns, errs = compare(name, r, c)
        if warns or errs:
            if errs:
                logger.error(f"{name}: Rust {r}")
                logger.error(f"{name}: CUDA {c}")
            else:
                logger.warning(f"{name}: Rust {r}")
                logger.warning(f"{name}: CUDA {c}")
            for warn_msg in warns:
                logger.warning(f"{name}: Rust != CUDA: {warn_msg}")
            for err_msg in errs:
                logger.error(f"{name}: Rust != CUDA: {err_msg}")
                ok = False
    elif r and not c:
        if name.startswith("cuda"):
            logger.warning(f"Rust declares {name}, CUDA missing -- accepting as looks like a libcuda function")
        else:
            logger.error(f"Rust declares {name}, CUDA missing"); ok = False
    elif c and not r:
        logger.info(f"CUDA defines {name}, Rust missing")
print("OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
