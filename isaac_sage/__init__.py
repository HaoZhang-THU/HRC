"""SAGE: safety-shielded, team-consistent training of hard oblique decision-tree
carriers for cooperative payload transport (Isaac Lab G1 cooperative-transport
environment).

Module map (paper section -> file):
    Sec. III-A  affine CBF rows + exact joint QP        safety.py
    Sec. III-B  hard oblique trees, ST routing, proposal tree_policy.py
    Sec. III-C  SAIL penalty map                        sail.py
    Eq. 33      team-context critic                     critic.py
    Sec. III-D  TALO update field                       talo.py
    Alg. 1 / 2  training loop / deployment controller   trainer.py
    Isaac Lab   environment adapter                     isaac_env.py
"""
from .critic import JointCritic, TeamContextCritic, ValueNorm
from .safety import FormationSafetyModel, RowSet, SafetyConfig, SafetyState
from .sail import sail_penalty_map
from .talo import talo_update_field
from .trainer import SAGEConfig, SAGETrainer, deploy_action, train
from .tree_policy import HardObliqueTree, TeamActor

__all__ = ["TeamContextCritic", "JointCritic", "ValueNorm", "FormationSafetyModel", "RowSet", "SafetyConfig", "SafetyState",
           "sail_penalty_map", "talo_update_field", "SAGEConfig", "SAGETrainer", "deploy_action",
           "train", "HardObliqueTree", "TeamActor"]
