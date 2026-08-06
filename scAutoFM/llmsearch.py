def exp_peft_search_prompt(arch_dict):
    prompt1 = "\nHere are some experimental results for PEFT configurations on the current task:\n"
    prompt2 = "\n#You must generate PEFT configurations that are strictly different from all the above results.#\n"

    # 排序
    arch_l = list(arch_dict.keys())  # 每个 key 是一个配置 tuple
    acc_l = [arch_dict[key] for key in arch_l]
    sorted_results = sorted(zip(arch_l, acc_l), key=lambda x: x[1], reverse=True)

    seen = set()
    operation_unique = []
    acc_unique = []
    operation_repeat = set()

    for arch, acc in sorted_results:
        if arch in seen:
            operation_repeat.add(arch)
        else:
            seen.add(arch)
            operation_unique.append(arch)
            acc_unique.append(acc)

    prompt_repeat = ""
    if len(operation_repeat) > 0:
        prompt_repeat = "The following configurations are repeated and should be avoided:\n" + \
                        ''.join([f"Config: {arch}\n" for arch in operation_repeat]) + \
                        "#Do not generate them again.#\n"

    prompt1 += ''.join([
        f"Config: {arch} -> Accuracy: {acc:.4f}\n" for arch, acc in zip(operation_unique, acc_unique)
    ])
    return prompt1 + prompt_repeat + prompt2


def prompt_peft_search(taskname, arch_dict=None, stage=0):
    prompt1 = f"""The task is to find the best PEFT configuration for the transformer model on the dataset `{taskname}`.
            The goal is to maximize accuracy with minimal trainable parameters.

            Each configuration includes:
            - `LORA DIM` (LoRA rank): one of [0, 5, 10, 50, 100]
            - `LORA DEPTH`: number of transformer layers with LoRA enabled: [3, 6, 9, 12]
            - `ADAPTER DIM`: one of [0, 5, 10, 50, 100]
            - `ADAPTER DEPTH`: one of [3, 6, 9, 12]
            - `PAEALLEL ADAPTER DIM`: one of [0, 5, 10, 50, 100]
            - `PAEALLEL ADAPTER DEPTH`: one of [3, 6, 9, 12]
            - `PREFIX DIM`: one of [0, 5, 10, 25, 50]
            - `PREFIX DEPTH`: one of [3, 6, 9, 12]

            Each configuration is a dictionary like:
            {
                "lora": [50, 50, 10, 50, 50, 50, 0, 0, 0, 0, 0, 0],
                "prefix": [10, 100, 10, 10, 0, 10, 10, 10, 10, 0, 0, 0],
                "adapter": [10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0],
                "parallel adapter": [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }.
            where each array has a fixed length of 12 corresponding to the 12 transformer layers, and the values indicate the dimension used by that module at each layer (0 means not used).
            """
    operation_prompt = """
        There are 5 operations that can be selected, including: lora, adapter, parallel adapter, prefix and skip.
        The define for lora is as follows:
        {
            The Low-Rank Adaptation (LoRA) operator introduces a pair of low-rank matrices $(A \\in \\mathbb{R}^{d \\times r}, B \\in \\mathbb{R}^{r \\times d})$ to approximate weight updates.
            $$\\Delta W = BA,$$
            and the adapted output is:
            $$\\mathbf{y} = W \\mathbf{x} + \\alpha \\cdot BA \\mathbf{x},$$
            where $\\alpha$ is a scaling factor and $r$ is the low-rank dimension.
        }

        The define for adapter is as follows:
        {
            The Adapter operator adds a bottleneck MLP module between transformer layers:
            $$\\text{Adapter}(\\mathbf{x}) = W_{\\text{up}} \\cdot \\sigma(W_{\\text{down}} \\cdot \\mathbf{x}),$$
            where $W_{\\text{down}} \\in \\mathbb{R}^{d \\times r}$, $W_{\\text{up}} \\in \\mathbb{R}^{r \\times d}$, and $r$ is the adapter dimension.
            The adapter output is added to the residual connection:
            $$\\mathbf{y} = \\mathbf{x} + \\text{Adapter}(\\mathbf{x}).$$
        }

        The define for parallel adapter is as follows:
        {
            The Parallel Adapter applies the same adapter module as standard adapter, but computes it in parallel to the main forward path, and sums the outputs:
            $$\\mathbf{y} = \\text{Main}(\\mathbf{x}) + \\text{Adapter}(\\mathbf{x}).$$
        }

        The define for prefix is as follows:
        {
            The Prefix Tuning operator prepends learnable continuous prefix vectors to the key and value matrices in attention:
            $$K' = [P_K; K], \quad V' = [P_V; V],$$
            where $P_K, P_V \\in \\mathbb{R}^{l_p \\times d}$ are learnable prefix embeddings of length $l_p$.
            The attention is computed over the extended sequence:
            $$\\text{Attention}(Q, K', V').$$
        }

        The define for skip is as follows:
        {
            The skip operation means this layer do not use any other PEFT modules, just outputs the main forward path:
            $$\\mathbf{y} = \\text{Main}(\\mathbf{x}).$$
        }
        """

    prompt2 = """
            You should generate 10 new configurations at a time.
            Each configuration must be different from all previous results.

            In the Exploration stage, explore all value combinations to understand which dimensions and depths are more effective.

            In the Exploitation stage, focus on refining top-performing configurations, especially those ranked in the top 20% of results.

            #You must not repeat any configuration that has already been tested.#

            Please strictly follow the format below:
            1. Config: {
                    "lora": [...],
                    "prefix": [...],
                    "adapter": [...],
                    "parallel adapter": [...]
                }
            2. ...
            10. Config: {...}
            """

    notice1 = "\n#We are in the Exploration stage. Please explore diverse configurations broadly.#\n"
    notice2 = "\n#We are in the Exploitation stage. Focus on optimizing top configurations and avoid low-performing settings.#\n"

    suffix = "#Return only 10 different configurations, nothing else.#"

    if stage == 0:
        return prompt1 + notice1 + prompt2 + suffix
    elif stage < 4:
        return prompt1 + exp_peft_search_prompt(arch_dict) + notice1 + prompt2 + suffix
    else:
        return prompt1 + exp_peft_search_prompt(arch_dict) + notice2 + prompt2 + suffix

class LLM4NAS():
    def __init__(self, search_space: SearchSpaceBase, config: dict, **kwargs):
        self.search_space = search_space
        self.trainer = trainer
        self.config = config
        self.dataname = self.config.dataname
        self.arch_dict = {}
        llm = self.config.llm
        if llm == "ChatGPT":
            self.llm = ChatGPT(self.config.api_key, self.config.llm_model)
        elif llm == "Qianfan":
            self.llm = Qianfan(self.config.api_key, self.config.secret_key, self.config.llm_model)
        else:
            print("this llm have not been achieve")

    # 生成prompt
    def gen_prompt(self, stage):
        if stage == 0:
            return prompt_peft_search(self.dataname, best_link(self.dataname), stage=stage)
        arch_dict = {}
        for key in self.arch_dict.keys():
            arch_dict[key] = self.arch_dict[key]
        return prompt_peft_search(self.dataname, best_link(self.dataname), arch_dict=arch_dict, stage=stage)

    def gen_response(self, prompt):
        system_content = '''Please pay special attention to my use of special markup symbols in the content below.The special markup symbols is # # ,and the content that needs special attention will be between #.'''
        history_chat = []
        if self.config.llm == "ChatGPT":
            response = self.llm.response(system_content, prompt)  # 获取llm的响应
        elif self.config.llm == "Qianfan":
            prompt = system + prompt
            response = self.llm.response(prompt)
        elif self.config.llm == "Llama":
            response,history = self.llm.response(prompt, history_chat)
            history_chat.append(history)
        return response

    # 处理llm生成的信息，并返回符合要求的操作列表（需要根据output_format来修改）
    def check_reponse(self, response):
        input_lst = response.split('Model:')
        archs = []
        for i in range(1, len(input_lst)):
            operations_str = input_lst[i].split('[')[1].split(']')[0]
            operations_list_str = operations_str.replace(" ", "")
            if operations_list_str == ['skip,skip,skip,skip']:
                continue
            archs.append(operations_list_str)
        return archs

    
    # 根据迭代生成prompt，并返回最优GNN, 默认迭代10次
    def fit(self, data) -> GNNBase:
        iterations = 2
        performance_history = [{'arch': 'gin,cheb,arma,graph', 'score': 0.2}]
        for iteration in range(iterations):
            prompt = self.gen_prompt(iteration)  # 获得prompt
            print(prompt)
            response = self.gen_response(prompt)  # 获取llm的响应
            archs = self.check_reponse(response)  # 根据response的字符规范化处理

            # 根据架构生成gnn，并进行评价
            for arch in archs:
                arch = arch.split(",")
                # print(type(arch))
                # print(arch)
                # arch = ["gcn", "gin"]
                gnn = self.search_space.to_gnn(arch)  # fitting SerchSpace needs to be done
                score = self.trainer.evaluate(data, gnn)  # fitting TrainerBase needs to be done
                performence = {
                    'arch': arch,
                    'score': score
                }
                performance_history.append(performence)
                self.arch_dict[arch] = self.trainer.get_result(score)

        # 获取最优架构
        print(performance_history)
        best_arch = max(performance_history, key=lambda x: x['score'])
        best_gnn = self.search_space.to_gnn(best_arch['arch'])
        return best_gnn

    def reset(self):
        self.llm = None
        self.best_model = None