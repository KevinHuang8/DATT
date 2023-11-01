# Get the directory of the current script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the parent directory of the script
PARENT_DIR="$(dirname "$DIR")"

# Add the parent directory to PYTHONPATH
export PYTHONPATH="$PARENT_DIR:$PYTHONPATH"
