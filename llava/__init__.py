import sys
import os
username = os.getlogin()

if username == 'eco':
    sys.path.append('/home/eco/CodeSpace/Venus')

from .model import LlavaLlamaForCausalLM
