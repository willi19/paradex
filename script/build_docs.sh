cd docs

echo "Building documentation..."
sphinx-build -b html . _build/html

echo "Cleaning up old files..."
rm -f *.html
rm -rf _sources _modules
# _static은 지우지 않음!

echo "Copying files to docs root..."
cp -r _build/html/* .
touch .nojekyll

echo "Removing _build directory..."
rm -rf _build
