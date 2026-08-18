import json
from openai import OpenAI


JUDGE_SYSTEM_PROMPT = (
    "You are an expert grader that determines if answers to questions "
    "match a gold standard answer"
)

JUDGE_USER_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:

Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.

Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


class AnswerJudge:
    def __init__(
        self,
        api_key: str,
        api_url: str,
        model: str,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_url,
        )
        self.model = model

    def judge(
        self,
        question: str,
        golden_answer: str,
        generated_answer: str,
    ) -> str:
        """
        判断 generated_answer 是否与 golden_answer 匹配。

        Returns:
            "CORRECT" or "WRONG"
        """

        user_prompt = JUDGE_USER_PROMPT.format(
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": JUDGE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
        )

        content = response.choices[0].message.content.strip()

        try:
            result = json.loads(content)
            label = result["label"].upper()

            if label not in ("CORRECT", "WRONG"):
                raise ValueError(f"Invalid label: {label}")

            return label

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(
                f"Judge 返回格式错误: {content}"
            ) from e