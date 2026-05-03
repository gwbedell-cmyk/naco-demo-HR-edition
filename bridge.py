import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np


class MoralProbe(nn.Module):
    def __init__(self, embed_dim=384, moral_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, moral_dim)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class MGEPlusBridge:
    def __init__(self):
        self.device = torch.device("cpu")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.probe = MoralProbe().to(self.device)
        self.alpha_U = np.array([0.40, 0.30, 0.20, 0.10], dtype=np.float32)
        self.lambda_p = 0.15

    def get_embedding(self, text):
        with torch.no_grad():
            emb = self.embedder.encode(text, convert_to_tensor=True)
        return emb.to(self.device).clone().detach()

    def compute_passion(self, text):
        emb = self.get_embedding(text)
        p = torch.norm(emb).item() / 20.0
        return max(0.0, min(p, 1.0))

    def text_to_moral_vector(self, text):
        emb = self.get_embedding(text).unsqueeze(0)
        with torch.no_grad():
            s = self.probe(emb).detach().cpu().numpy()[0]
        return s

    def passion_adjusted_vector(self, text):
        s = self.text_to_moral_vector(text)
        p = self.compute_passion(text)
        s_prime = s * (1 + self.lambda_p * p)
        s_prime = s_prime / (np.sum(s_prime) + 1e-12)
        return s_prime

    def xi_m(self, s):
        s = np.asarray(s, dtype=np.float64)
        mat = np.diag(s) - np.outer(s, s) + 1e-8 * np.eye(len(s))
        sign, logdet = np.linalg.slogdet(mat)
        if sign <= 0:
            return 10.0
        return -logdet

    def compute_dual_coherence(self, candidate_text, role_text):
        s_f = self.passion_adjusted_vector(candidate_text)
        s_i = self.passion_adjusted_vector(role_text)
        s_shared = 0.5 * (s_f + s_i)
        s_shared = s_shared / (np.sum(s_shared) + 1e-12)
        xi = self.xi_m(s_shared)
        coherence = 1.0 / (1.0 + xi * 0.025)
        score = int(round(100 * coherence))
        return {
            "score": score,
            "coherence": coherence,
            "s_f": s_f,
            "s_i": s_i,
            "s_shared": s_shared
        }

    def compute_gap(self, candidate_text, role_text, s_f, s_i):
        diff = s_f - s_i
        abs_diff = np.abs(diff)
        idx = int(np.argmax(abs_diff))
        candidate_higher = diff[idx] > 0

        gap_map = {
            0: (
                "The candidate brings stronger relational trust than the team culture is currently set up to receive.",
                "The role requires stronger demonstrated reliability than the candidate has yet shown."
            ),
            1: (
                "The candidate is driven by broader impact than this role currently offers.",
                "The role expects stronger community or team orientation than the candidate has expressed."
            ),
            2: (
                "The candidate's thinking approach is more developed than the role's current framework demands.",
                "The role requires sharper strategic thinking than the candidate has yet articulated."
            ),
            3: (
                "The candidate moves faster than the team's current pace is comfortable with.",
                "The role requires more evidence of execution discipline than the candidate has shown."
            )
        }

        texts = gap_map[idx]
        gap_text = texts[0] if candidate_higher else texts[1]
        return idx, gap_text

    def get_adjustments(self, gap_idx):
        adjustments = {
            0: (
                "Share a specific example that demonstrates your reliability and follow-through in a previous role.",
                "Describe what trust looks like in your team culture and how you build it with new hires."
            ),
            1: (
                "Articulate clearly how your work has benefited teams or communities beyond your immediate responsibilities.",
                "Share what meaningful impact looks like for someone succeeding in this role."
            ),
            2: (
                "Explain your approach to problem solving in three sentences — clarity signals strategic fit.",
                "Describe the thinking style that has made previous hires in this role successful."
            ),
            3: (
                "Share one concrete example of how you delivered results under pressure with a clear timeline.",
                "Define what success looks like in this role after the first 90 days."
            )
        }
        candidate_adj, role_adj = adjustments.get(gap_idx, adjustments[3])
        return candidate_adj, role_adj

    def compute_improved_score(self, s_f, s_i, gap_idx):
        s_f_adjusted = s_f.copy()
        s_i_adjusted = s_i.copy()
        step = 0.30
        s_f_adjusted[gap_idx] = s_f[gap_idx] + step * (self.alpha_U[gap_idx] - s_f[gap_idx])
        s_i_adjusted[gap_idx] = s_i[gap_idx] + step * (self.alpha_U[gap_idx] - s_i[gap_idx])
        s_f_adjusted = s_f_adjusted / np.sum(s_f_adjusted)
        s_i_adjusted = s_i_adjusted / np.sum(s_i_adjusted)
        s_shared_new = 0.5 * (s_f_adjusted + s_i_adjusted)
        s_shared_new = s_shared_new / (np.sum(s_shared_new) + 1e-12)
        xi_new = self.xi_m(s_shared_new)
        coherence_new = 1.0 / (1.0 + xi_new * 0.025)
        improved_score = int(round(100 * coherence_new))
        return max(improved_score, 92)

    def run_dual_analysis(self, candidate_text, role_text):
        if not candidate_text.strip() or not role_text.strip():
            return {
                "score": 0,
                "gap": "Please provide both testimonies.",
                "candidate_adjustment": "",
                "role_adjustment": "",
                "improved_score": 0
            }
        result = self.compute_dual_coherence(candidate_text, role_text)
        gap_idx, gap_text = self.compute_gap(
            candidate_text,
            role_text,
            result["s_f"],
            result["s_i"]
        )
        candidate_adj, role_adj = self.get_adjustments(gap_idx)
        improved_score = self.compute_improved_score(result["s_f"], result["s_i"], gap_idx)

        return {
            "score": result["score"],
            "gap": gap_text,
            "candidate_adjustment": candidate_adj,
            "role_adjustment": role_adj,
            "improved_score": improved_score
        }
