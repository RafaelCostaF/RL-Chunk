import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy

nlp = spacy.load("en_core_web_md")

MAX_SELECTED_CHUNKS = 10

def chunk_text(text, max_length=500):
    """Split text into chunks of approximately `max_length` characters."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks[:]

class FixedChunkEnvGranularReward(gym.Env):
    def __init__(self, row, inference_mode=False, print_info=False, continuous_action=False):
        super().__init__()
        self.row = row
        self.inference_mode = inference_mode
        self.print_info = print_info
        self.continuous_action = continuous_action

        self.observation_space = spaces.Dict({
            "similarity": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "remaining_budget": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        if self.continuous_action:
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.query = self.row['query']
        self.answer = self.row['answer']
        if self.print_info:
            print(f"Query: {self.query}")
            print(f"Answer: {self.answer}")
        self.chunks = chunk_text(self.row['page_results_text'])
        self.query_doc = nlp(self.query)

        self.selected_chunks = []
        self.current_chunk_idx = 0
        self.total_reward = 0.0
        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        done = False
        reward = 0.0
        truncated = False

        if self.continuous_action:
            action = int(action[0] > 0.5)

        chunk = self.chunks[self.current_chunk_idx]
        similarity = self.query_doc.similarity(nlp(chunk))

        if action == 1:
            self.selected_chunks.append(chunk)
            reward = self._intermediate_reward(similarity)

        self.total_reward += reward
        self.step_count += 1
        self.current_chunk_idx += 1

        if (
            self.current_chunk_idx >= len(self.chunks)
            or len(self.selected_chunks) >= MAX_SELECTED_CHUNKS
        ):
            done = True

        info = {}
        if done:
            info["episode"] = {
                "r": float(self.total_reward),
                "l": int(self.step_count)
            }

        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        if self.current_chunk_idx >= len(self.chunks):
            return {
                "similarity": np.float32(0.0),
                "remaining_budget": np.float32(0.0),
            }

        chunk = self.chunks[self.current_chunk_idx]
        sim = self.query_doc.similarity(nlp(chunk))
        remaining_budget = MAX_SELECTED_CHUNKS - len(self.selected_chunks)

        return {
            "similarity": np.float32(sim),
            "remaining_budget": np.float32(remaining_budget / MAX_SELECTED_CHUNKS),
        }

    def _intermediate_reward(self, sim):
        if sim < 0.2:
            return -1.0  # muito ruim
        elif sim < 0.4:
            return -0.5  # ruim
        elif sim < 0.5:
            return 0.0   # neutro negativo
        elif sim < 0.6:
            return 0.5   # neutro positivo
        elif sim < 0.75:
            return 1.5   # bom
        elif sim < 0.9:
            return 2.5   # muito bom
        elif sim < 0.93:
            return 4.0   # excelente
        elif sim < 0.95:
            return 5.0   # excelente+
        elif sim < 0.97:
            return 6.5   # quase perfeito
        elif sim < 0.98:
            return 8.0   # muito próximo do ideal
        elif sim < 0.99:
            return 9.0   # excelente extremo
        else:
            return 10.0  # perfeito
