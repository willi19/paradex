if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <system_name>"
    exit 1
fi

SYSTEM_NAME="$1"

cp -r "config/${SYSTEM_NAME}/." config/current