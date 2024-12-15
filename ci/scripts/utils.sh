install_s5cmd() {
    arch=$(uname -m)
    case $arch in
    arm64|aarch64)
        S5CMD_BIN="s5cmd_2.2.2_linux_arm64.deb"
        ;;
    x86_64|amd64)
        S5CMD_BIN="s5cmd_2.2.2_linux_amd64.deb"
        ;;
    *)
        echo "Unsupported architecture: $arch"
        exit 1
        ;;
    esac

    echo "Checking s5cmd"
    if type s5cmd &>/dev/null; then
        echo "s5cmd was installed."
    else
        TMP_DIR=/tmp/s5cmd
        rm -rf $TMP_DIR
        mkdir $TMP_DIR
        echo "s5cmd was not installed. Installing.."
        wget "https://github.com/peak/s5cmd/releases/download/v2.2.2/${S5CMD_BIN}" -P $TMP_DIR
        sudo dpkg -i "${TMP_DIR}/${S5CMD_BIN}"
    fi
}
