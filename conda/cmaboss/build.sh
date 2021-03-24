cd engine/src
make grammars
cd ../python
cp -r ../src .
$PYTHON setup.py install

