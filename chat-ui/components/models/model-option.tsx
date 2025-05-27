import { LLM } from "@/types"
import { FC } from "react"
import { ModelIcon } from "./model-icon"
import { IconInfoCircle } from "@tabler/icons-react"
import { WithTooltip } from "../ui/with-tooltip"

interface ModelOptionProps {
  model: LLM
  onSelect: () => void
}

export const ModelOption: FC<ModelOptionProps> = ({ model, onSelect }) => {
  const isOffline = (model as any).status === 'offline';
  return (
    <WithTooltip
      display={
        <div>
          {isOffline && (
            <div className="text-red-500 text-xs font-semibold">
              {model.description || 'Model is offline or not available on this node.'}
            </div>
          )}
          {model.provider !== "ollama" && (model as any).pricing && (
            <div className="space-y-1 text-sm mt-1">
              <div>
                <span className="font-semibold">Input Cost:</span>{" "}
                {(model as any).pricing.inputCost} {(model as any).pricing.currency} per{" "}
                {(model as any).pricing.unit}
              </div>
              {(model as any).pricing.outputCost && (
                <div>
                  <span className="font-semibold">Output Cost:</span>{" "}
                  {(model as any).pricing.outputCost} {(model as any).pricing.currency} per{" "}
                  {(model as any).pricing.unit}
                </div>
              )}
            </div>
          )}
        </div>
      }
      side="bottom"
      trigger={
        <div
          className={`flex w-full justify-start space-x-3 truncate rounded p-2 ${isOffline ? 'bg-gray-200 text-gray-400 cursor-not-allowed opacity-60' : 'hover:bg-accent cursor-pointer hover:opacity-50'}`}
          onClick={isOffline ? undefined : onSelect}
          style={isOffline ? { pointerEvents: 'none' } : {}}
        >
          <div className="flex items-center space-x-2">
            <ModelIcon provider={model.provider} width={28} height={28} />
            <div className="text-sm font-semibold">{model.modelName}</div>
            {isOffline && (
              <span className="ml-2 text-xs text-red-500 font-bold">OFFLINE</span>
            )}
          </div>
        </div>
      }
    />
  )
}

