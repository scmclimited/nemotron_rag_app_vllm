#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/scmclimited/deep_rag"
WORKDIR="$(pwd)"
TMP_DIR="$WORKDIR/.tmp_deep_rag"

echo "[*] Cloning deep_rag into $TMP_DIR ..."
rm -rf "$TMP_DIR"
git clone --depth 1 "$REPO_URL" "$TMP_DIR"

echo "[*] Copying vector_db/ and deep_rag_frontend_vue/ ..."
rm -rf "$WORKDIR/vector_db" "$WORKDIR/deep_rag_frontend_vue"
cp -R "$TMP_DIR/vector_db" "../$WORKDIR/vector_db"
cp -R "$TMP_DIR/deep_rag_frontend_vue" "../$WORKDIR/deep_rag_frontend_vue"

echo "[*] Optionally copy md_guides/ and .env.example if needed ..."
if [ -d "$TMP_DIR/md_guides" ]; then
  cp -R "$TMP_DIR/md_guides" "../$WORKDIR/md_guides"
fi

echo "[*] Cleaning up ..."
rm -rf "$TMP_DIR"

echo "[*] Done. You can now configure and build the frontend and vector DB."
