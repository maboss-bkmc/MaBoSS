$PYTHON -m pip install engine/python --no-deps --ignore-installed --no-cache-dir -vvv
mkdir -p ${PREFIX}/tests
cp -r engine/tests/* ${PREFIX}/tests

