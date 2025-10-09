cd docs
make clean
make html
cp -r _build/html/* .
rm -rf _build