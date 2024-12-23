import AnthropicBedrock from "@anthropic-ai/bedrock-sdk"
import { Anthropic } from "@anthropic-ai/sdk"
import { ApiHandler } from "../"
import { ApiHandlerOptions, bedrockDefaultModelId, BedrockModelId, bedrockModels, ModelInfo } from "../../shared/api"
import { ApiStream } from "../transform/stream"
import {  BedrockRuntime } from 'aws-sdk'
import { InvokeModelCommand, InvokeModelWithResponseStreamCommand, BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime";
import { Readable } from "stream";

// https://docs.anthropic.com/en/api/claude-on-amazon-bedrock
export class AwsBedrockHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: AnthropicBedrock
  private bedrock: BedrockRuntimeClient;

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.client = new AnthropicBedrock({
			// Authenticate by either providing the keys below or use the default AWS credential providers, such as
			// using ~/.aws/credentials or the "AWS_SECRET_ACCESS_KEY" and "AWS_ACCESS_KEY_ID" environment variables.
			...(this.options.awsAccessKey ? { awsAccessKey: this.options.awsAccessKey } : {}),
			...(this.options.awsSecretKey ? { awsSecretKey: this.options.awsSecretKey } : {}),
			...(this.options.awsSessionToken ? { awsSessionToken: this.options.awsSessionToken } : {}),

			// awsRegion changes the aws region to which the request is made. By default, we read AWS_REGION,
			// and if that's not present, we default to us-east-1. Note that we do not read ~/.aws/config for the region.
			awsRegion: this.options.awsRegion,
		})
      this.bedrock = new BedrockRuntimeClient({
			...(this.options.awsAccessKey ? { credentials: {accessKeyId: this.options.awsAccessKey, secretAccessKey: this.options.awsSecretKey || "", sessionToken: this.options.awsSessionToken || ""} } : {}),
			region: this.options.awsRegion,
		});
	}

	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
        const model = this.getModel();
		// cross region inference requires prefixing the model id with the region
		let modelId: string
		if (this.options.awsUseCrossRegionInference) {
			let regionPrefix = (this.options.awsRegion || "").slice(0, 3)
			switch (regionPrefix) {
				case "us-":
					modelId = `us.${model.id}`
					break
				case "eu-":
					modelId = `eu.${model.id}`
					break
				default:
					// cross region inference is not supported in this region, falling back to default model
					modelId = model.id
					break
			}
		} else {
			modelId = model.id
		}

        if (model.id.startsWith("anthropic")) {
		const stream = await this.client.messages.create({
			model: modelId,
			max_tokens: model.info.maxTokens || 8192,
			temperature: 0,
			system: systemPrompt,
			messages,
			stream: true,
		})
		for await (const chunk of stream) {
			switch (chunk.type) {
				case "message_start":
					const usage = chunk.message.usage
					yield {
						type: "usage",
						inputTokens: usage.input_tokens || 0,
						outputTokens: usage.output_tokens || 0,
					}
					break
				case "message_delta":
					yield {
						type: "usage",
						inputTokens: 0,
						outputTokens: chunk.usage.output_tokens || 0,
					}
					break

				case "content_block_start":
					switch (chunk.content_block.type) {
						case "text":
							if (chunk.index > 0) {
								yield {
									type: "text",
									text: "\n",
								}
							}
							yield {
								type: "text",
								text: chunk.content_block.text,
							}
							break
					}
					break
				case "content_block_delta":
					switch (chunk.delta.type) {
						case "text_delta":
							yield {
								type: "text",
								text: chunk.delta.text,
							}
							break
					}
					break
			}
		}
     }
        else if (model.id.startsWith("amazon")) {
            const command = new InvokeModelWithResponseStreamCommand({
                modelId: modelId,
                contentType: "application/json",
                accept: "application/json",
                body: JSON.stringify({
					messages: messages,
					temperature: 0,
					max_tokens: model.info.maxTokens || 8192,
				}),
            });
            const response = await this.bedrock.send(command);
             if (response.body) {
                for await (const event of response.body) {

                    if(event.chunk) {
                        const responseBody = JSON.parse(new TextDecoder().decode(event.chunk.bytes));
                         const usage = responseBody.usage;
                        if (usage) {
                             yield {
                                type: "usage",
                                inputTokens: usage.inputTokens || 0,
                                outputTokens: usage.outputTokens || 0
                             }
                         }
                         if(responseBody.messages) {
                              for(const message of responseBody.messages) {
                                    yield {
                                       type: "text",
                                        text: message.content
                                    }
                                }
                         } else if(responseBody.completion) {
                                yield {
                                     type: "text",
                                     text: responseBody.completion
                                }
                            }
                    }
                 }
             }
        }
	}

    async *invokeModel(prompt: string): ApiStream {
        const model = this.getModel();
        let modelId: string
		if (this.options.awsUseCrossRegionInference) {
			let regionPrefix = (this.options.awsRegion || "").slice(0, 3)
			switch (regionPrefix) {
				case "us-":
					modelId = `us.${model.id}`
					break
				case "eu-":
					modelId = `eu.${model.id}`
					break
				default:
					// cross region inference is not supported in this region, falling back to default model
					modelId = model.id
					break
			}
		} else {
			modelId = model.id
		}
        
        if (model.id.startsWith("amazon")) {
            const command = new InvokeModelCommand({
                 modelId: modelId,
                 contentType: "application/json",
                 accept: "application/json",
                 body: JSON.stringify({
                     prompt: prompt,
                     temperature: 0,
                     max_tokens: model.info.maxTokens || 8192,
                 }),
            })
            const response = await this.bedrock.send(command);
            const responseBody = JSON.parse(new TextDecoder().decode(response.body));
            const usage = responseBody.usage;

             if(usage) {
                  yield {
                     type: "usage",
                     inputTokens: usage.inputTokens || 0,
                     outputTokens: usage.outputTokens || 0
                }
            }

           yield {
               type: "text",
                text: responseBody.completion
            }
        }
    }

	getModel(): { id: BedrockModelId; info: ModelInfo } {
		const modelId = this.options.apiModelId
		if (modelId && modelId in bedrockModels) {
			const id = modelId as BedrockModelId
			return { id, info: bedrockModels[id] }
		}
		return { id: bedrockDefaultModelId, info: bedrockModels[bedrockDefaultModelId] }
	}
}