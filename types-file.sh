#!/bin/bash

# Default values
SEARCH_DIR="."
OUTPUT_FILE="combined.d.ts"

# Parse command line arguments
if [ "$#" -ge 1 ]; then
  SEARCH_DIR="$1"
fi

if [ "$#" -ge 2 ]; then
  OUTPUT_FILE="$2"
fi

# Check if the search directory exists
if [ ! -d "$SEARCH_DIR" ]; then
  echo "Error: Directory '$SEARCH_DIR' does not exist."
  exit 1
fi

# Create combined file with headers
{
  echo "// Combined TypeScript declarations from $SEARCH_DIR"
  echo "// Generated on $(date)"
  
  # Process files one by one to avoid command line length limits
  find "$SEARCH_DIR" -name "*.d.ts" -type f | sort | while read -r file; do
    echo -e "\n// ============================================================================"
    echo "// Source: $file"
    echo "// ============================================================================"
    echo ""
    cat "$file"
  done
} > "$OUTPUT_FILE"

# Count the files that were combined
FILE_COUNT=$(grep -c "// Source:" "$OUTPUT_FILE")

echo "Successfully combined $FILE_COUNT .d.ts files from '$SEARCH_DIR' into '$OUTPUT_FILE'"
echo "Output file size: $(du -h "$OUTPUT_FILE" | cut -f1)"