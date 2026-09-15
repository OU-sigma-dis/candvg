#!/bin/bash
set -euo pipefail

IO_ROOT=${1:?"Usage: $0 path/to/batches"}
INPUT_FILE_PATH="$IO_ROOT/batch_input.jsonl"
OUTPUT_FILE_PATH="$IO_ROOT/batch_output.jsonl"
BATCH_ID_FILE="$IO_ROOT/batch_id.txt"

: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
[ -f "$INPUT_FILE_PATH" ] || { echo "Input file not found: $INPUT_FILE_PATH" >&2; exit 1; }

json_value() {
  python3 -c '
import json
import sys

data = json.load(sys.stdin)
value = data.get(sys.argv[1])
if value is None:
    error = data.get("error") or {}
    message = error.get("message", f"Missing JSON field: {sys.argv[1]}")
    raise SystemExit(message)
print(value)
' "$1"
}

if [ ! -f "$BATCH_ID_FILE" ]; then
  echo "Uploading $INPUT_FILE_PATH..."
  RESPONSE=$(curl -sS https://api.openai.com/v1/files \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -F purpose=batch \
    -F file=@"$INPUT_FILE_PATH")
  FILE_ID=$(json_value id <<< "$RESPONSE")

  echo "$FILE_ID" > "$IO_ROOT/file_id.txt"
  echo "Uploaded file_id: $FILE_ID"

  echo "Submitting batch job..."
  RESPONSE=$(curl -sS https://api.openai.com/v1/batches \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
      \"input_file_id\": \"$FILE_ID\",
      \"endpoint\": \"/v1/responses\",
      \"completion_window\": \"24h\"
    }")
  BATCH_ID=$(json_value id <<< "$RESPONSE")
  echo "$BATCH_ID" > "$BATCH_ID_FILE"
  echo "Submitted batch_id: $BATCH_ID"
fi

BATCH_ID=$(cat "$BATCH_ID_FILE")

echo "Waiting for batch to complete..."
while true; do
  RESPONSE=$(curl -sS "https://api.openai.com/v1/batches/$BATCH_ID" \
    -H "Authorization: Bearer $OPENAI_API_KEY")

  STATUS=$(json_value status <<< "$RESPONSE")

  case "$STATUS" in
    completed) break ;;
    failed|expired|cancelled)
      echo "Batch ended with status: $STATUS" >&2
      exit 1
      ;;
  esac
  sleep 60
done

echo "Fetching output_file_id..."
RESPONSE=$(curl -sS "https://api.openai.com/v1/batches/$BATCH_ID" \
  -H "Authorization: Bearer $OPENAI_API_KEY")
OUTPUT_FILE_ID=$(json_value output_file_id <<< "$RESPONSE")
echo "$OUTPUT_FILE_ID" > "$IO_ROOT/output_file_id.txt"
echo "Output file ID: $OUTPUT_FILE_ID"

echo "Downloading output to $OUTPUT_FILE_PATH"
curl -sS "https://api.openai.com/v1/files/$OUTPUT_FILE_ID/content" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  --output "$OUTPUT_FILE_PATH"

echo "Done. Output saved to $OUTPUT_FILE_PATH"
