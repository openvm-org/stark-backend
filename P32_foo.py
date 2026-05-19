a = [
    0x1bb8e645ae216da7,
    0x53fe3ab1e35c59e3,
    0x8c49833d53bb8085,
    0x0216d0b17f4e44a5,
]

for i in a:
    print(i % (1 << 32))
    print(i // (1 << 32))
