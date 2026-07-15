set -euo pipefail

SPHINX_BUILD="${SPHINX_BUILD:-sphinx-build}"
if [[ "${SPHINX_BUILD}" == "sphinx-build" && -x "${HOME}/anaconda3/envs/docs/bin/sphinx-build" ]]; then
  SPHINX_BUILD="${HOME}/anaconda3/envs/docs/bin/sphinx-build"
fi

cd docs

echo "Building documentation..."
"${SPHINX_BUILD}" -b html . _build/html

echo "Cleaning up old files..."
rm -f *.html
rm -rf _sources _modules autoapi
# _static은 지우지 않음!

echo "Copying files to docs root..."
cp -r _build/html/* .
touch .nojekyll

echo "Removing _build directory..."
rm -rf _build
