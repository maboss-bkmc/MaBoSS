cd engine/src
make 
cd ../python
cp -r ../src .
python setup.py build
python setup.py install --prefix=$PREFIX

