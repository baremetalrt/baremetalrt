import { ChatbotUIContext } from "@/context/context"
import { LLM, LLMID, ModelProvider } from "@/types"
import { IconCheck, IconChevronDown } from "@tabler/icons-react"
import { FC, useContext, useEffect, useRef, useState } from "react"
import { Button } from "../ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger
} from "../ui/dropdown-menu"
import { Input } from "../ui/input"
import { Tabs, TabsList, TabsTrigger } from "../ui/tabs"
import { ModelIcon } from "./model-icon"
import { ModelOption } from "./model-option"
import { LLM_LIST } from "@/lib/models/llm/llm-list"

interface ModelSelectProps {
  selectedModelId: string
  onSelectModel: (modelId: LLMID) => void
}

export const ModelSelect: FC<ModelSelectProps> = ({
  selectedModelId,
  onSelectModel
}) => {
  const {
    profile,
    models,
    availableHostedModels,
    availableLocalModels,
    availableOpenRouterModels
  } = useContext(ChatbotUIContext)

  const inputRef = useRef<HTMLInputElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)

  const [isOpen, setIsOpen] = useState(false)
  const [search, setSearch] = useState("")
  const [tab, setTab] = useState<"hosted" | "local">("hosted")

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => {
        inputRef.current?.focus()
      }, 100) // FIX: hacky
    }
  }, [isOpen])

  const handleSelectModel = (modelId: LLMID, isOffline = false) => {
    if (isOffline) return;
    if (modelId === selectedModelId) return; // Prevent redundant reload/warmup
    onSelectModel(modelId)
    setIsOpen(false)
  }

  // Merge backend status with frontend model list by modelId
  const backendStatusMap = Object.fromEntries(models.map((m: any) => [m.id, m.status]))
  type LLMWithStatus = LLM & { status?: string };
  const allModels: LLMWithStatus[] = [
    ...LLM_LIST.map((model: LLM) => ({
      ...model,
      status: backendStatusMap[model.modelId] || "offline"
    }))
  ]

  const hasStatus = (model: LLMWithStatus): model is LLMWithStatus & { status: string } => {
    return 'status' in model && model.status !== undefined;
  }

  const groupedModels = allModels.reduce<Record<string, LLMWithStatus[]>>(
    (groups, model) => {
      const key = model.provider
      if (!groups[key]) {
        groups[key] = []
      }
      groups[key].push(model)
      return groups
    },
    {}
  )

  // Find the model marked as online by the backend
  const backendOnlineModel = allModels.find(model => backendStatusMap[model.modelId] === 'online');
  let selectedModel = allModels.find(model => model.modelId === selectedModelId);
// If no model is selected, or the selected model is offline, auto-select the backend online model
useEffect(() => {
  if (!selectedModel || (hasStatus(selectedModel) && selectedModel.status === 'offline')) {
    if (backendOnlineModel && selectedModelId !== backendOnlineModel.modelId) {
      onSelectModel(backendOnlineModel.modelId);
    }
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [models, selectedModelId]);

  if (!profile) return null

  return (
    <DropdownMenu
      open={isOpen}
      onOpenChange={isOpen => {
        setIsOpen(isOpen)
        setSearch("")
      }}
    >
      <DropdownMenuTrigger
        className="bg-background w-full justify-start border-2 px-3 py-5"
        asChild
        disabled={allModels.length === 0}
      >
        {allModels.length === 0 ? (
          <div className="rounded text-sm font-bold">
            Unlock models by entering API keys in your profile settings.
          </div>
        ) : (
          <Button
            ref={triggerRef}
            className="flex items-center justify-between"
            variant="ghost"
          >
            <div className="flex items-center">
              {selectedModel ? (
                <>
                  <ModelIcon
                    provider={selectedModel?.provider}
                    width={26}
                    height={26}
                  />
                  <div className="ml-2 flex items-center">
                    {selectedModel?.modelName}
                  </div>
                </>
              ) : (
                <div className="flex items-center">Select a model</div>
              )}
            </div>

            <IconChevronDown />
          </Button>
        )}
      </DropdownMenuTrigger>

      <DropdownMenuContent
        className="space-y-2 overflow-auto p-2"
        style={{ width: triggerRef.current?.offsetWidth }}
        align="start"
      >
        <Tabs value={tab} onValueChange={(value: any) => setTab(value)}>
          {availableLocalModels.length > 0 && (
            <TabsList defaultValue="hosted" className="grid grid-cols-2">
              <TabsTrigger value="hosted">Hosted</TabsTrigger>

              <TabsTrigger value="local">Local</TabsTrigger>
            </TabsList>
          )}
        </Tabs>

        <Input
          ref={inputRef}
          className="w-full"
          placeholder="Search models..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />

        <div className="max-h-[300px] overflow-auto">
          {Object.entries(groupedModels).map(([provider, models]) => {
            const filteredModels = (models as LLMWithStatus[])
              .filter(model => {
                if (tab === "hosted") return model.provider !== "ollama"
                if (tab === "local") return model.provider === "ollama"
                if (tab === "openrouter") return model.provider === "openrouter"
              })
              .filter(model =>
                model.modelName.toLowerCase().includes(search.toLowerCase())
              )
              .sort((a, b) => a.provider.localeCompare(b.provider))

            if (filteredModels.length === 0) return null

            return (
              <div key={provider}>
                <div className="mb-1 ml-2 text-xs font-bold tracking-wide opacity-50">
                  {provider === "openai" && profile.use_azure_openai
                    ? "AZURE OPENAI"
                    : provider.toLocaleUpperCase()}
                </div>

                <div className="mb-4">
                  {filteredModels.map(model => {
  const isOffline = hasStatus(model) ? model.status === 'offline' : false;
  return (
    <div
      key={model.modelId}
      className="flex items-center space-x-1"
    >
      {selectedModelId === model.modelId && !isOffline && (
        <IconCheck className="ml-2" size={32} />
      )}
      <ModelOption
        key={model.modelId}
        model={model}
        onSelect={() => handleSelectModel(model.modelId, isOffline)}
        selected={selectedModelId === model.modelId && !isOffline}
      />
      {/* Status indicator */}
      <span
        className={`ml-2 text-xs font-bold ${
          model.status === 'offline' || model.status === 'warming_up'
            ? 'text-red-500'
            : 'text-green-600'
        }`}
      >
        {model.status === 'offline'
          ? 'OFFLINE'
          : model.status === 'warming_up'
          ? 'WARMING UP'
          : 'ONLINE'}
      </span>
    </div>
  );
})}
                </div>
              </div>
            )
          })}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
