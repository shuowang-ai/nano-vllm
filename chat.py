import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=512)

    messages = []
    print("Chat with Qwen3-0.6B (输入 'quit' 退出, 'clear' 清空历史)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.lower() == "clear":
            messages.clear()
            print("(历史已清空)\n")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = llm.generate([prompt], sampling_params)
        reply = outputs[0]["text"]

        # strip think block if present
        display = reply
        if "<think>" in display and "</think>" in display:
            display = display.split("</think>", 1)[1].strip()
        # remove trailing special tokens
        display = display.replace("<|im_end|>", "").strip()

        print(f"AI: {display}\n")
        messages.append({"role": "assistant", "content": reply.replace("<|im_end|>", "").strip()})


if __name__ == "__main__":
    main()
