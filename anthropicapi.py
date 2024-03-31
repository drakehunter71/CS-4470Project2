import anthropic
from apikeys import get_api_key


class AnthropicManager:
    def __init__(self):
        self.chat_history = []
        try:
            self.client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
        except:
            exit("Ooops! You forgot to set ANTHROPIC_API_KEY in your environment!")

    def chat(self, prompt="", model_name="claude-3-haiku-20240307"):
        if not prompt:
            print("Didn't receive input!")
            return

        chat_question = [{"role": "user", "content": prompt}]
        response = self.client.messages.create(
            model=model_name, max_tokens=1000, temperature=0, messages=chat_question
        )
        response_text = response.content[0].text
        return response_text
