$PYTHON -m pip install engine/python --no-deps --ignore-installed --no-cache-dir -vvv
mkdir -p ${PREFIX}/tests/cmaboss
cp -r engine/tests/cmaboss/* ${PREFIX}/tests/cmaboss

