#!/bin/bash
# Enhanced backend launcher with interactive model selection for TRTLLM

# Activate venv (edit path if your venv is elsewhere)
source ~/trtllm-venv/bin/activate

cd /mnt/c/Github/baremetalrt

# List of available models
# Canonical list of available models for backend selection
MODELS=(
  "llama3.1_8b_trtllm_instruct"  # INT8: /mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-engine
  "llama3.1_8b_trtllm_instruct_int4"  # INT4: /mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-engine
  "llama2_7b_chat_8int"
)
# NOTE: Ensure backend and engine directory are in sync for instruct model!

echo "Select a model to set as 'online':"
for i in "${!MODELS[@]}"; do
  echo "  $i: ${MODELS[$i]}"
done

read -p "Enter model number (0-${#MODELS[@]}-1): " CHOICE

if [[ "$CHOICE" =~ ^[0-9]+$ ]] && [ "$CHOICE" -ge 0 ] && [ "$CHOICE" -lt "${#MODELS[@]}" ]; then
  # Build JSON with selected model online, others offline
  JSON="{"
  for i in "${!MODELS[@]}"; do
    STATUS="offline"
    if [ "$i" -eq "$CHOICE" ]; then
      STATUS="online"
    fi
    JSON+="\"${MODELS[$i]}\": \"$STATUS\""
    if [ "$i" -lt $((${#MODELS[@]}-1)) ]; then
      JSON+=", "
    fi
  done
  JSON+="}"

  # Write to model_status.json
  echo "$JSON" > api/model_status.json
  echo "Set ${MODELS[$CHOICE]} as online in api/model_status.json."
else
  echo "Invalid selection. Exiting."
  exit 1
fi

# Start backend
uvicorn api.openai_api:app --host 0.0.0.0 --port 8000
