"""
Hypothesis generation module.
Extracted from agent.py - Phase 1: Idea Generation.
"""
import json
import os
import instructor
import litellm
from pydantic import BaseModel, ValidationError
from cellvoyager.utils import get_documentation

litellm.drop_params = True  # ignore unsupported params per-model silently

# Instructor client wrapping LiteLLM — handles retries, validation, and structured output
# for all OpenAI and Anthropic models uniformly.
_instructor_client = instructor.from_litellm(litellm.completion)
# JSON-mode client for providers that lack function-calling / tools support (e.g. Ollama).
_instructor_client_json = instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON)


_MODEL_ALIASES = {
    "gpt-5.3": "openai/gpt-5.3-chat-latest",
    "gpt-5.2": "openai/gpt-5.2-chat-latest",
}


def _normalize_model_name(model: str) -> str:
    """Add provider prefix for litellm if not already present."""
    if model in _MODEL_ALIASES:
        return _MODEL_ALIASES[model]
    if "/" in model:
        return model  # already has provider prefix
    if model.startswith("claude-") or model.startswith("anthropic"):
        return model  # litellm auto-detects Anthropic models
    # Known auto-detected OpenAI models
    _auto_detected = {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", "o1", "o3-mini", "o3", "o4-mini"}
    if model in _auto_detected:
        return model
    # For newer OpenAI models add the prefix
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return f"openai/{model}"
    return model


class AnalysisPlan(BaseModel):
    hypothesis: str
    analysis_plan: list[str]
    first_step_code: str
    code_description: str = ""
    summary: str = ""


class HypothesisGenerator:
    """
    Generates and refines analysis hypotheses/ideas.
    Called during the idea generation phase before execution.
    """

    def __init__(
        self,
        model_name,
        prompt_dir,
        coding_guidelines,
        coding_system_prompt,
        adata_summary,
        paper_summary,
        logger,
        use_self_critique=True,
        use_documentation=True,
        max_iterations=6,
        deepresearch_background="",
        log_prompts=False,
        client=None,  # kept for backward compat, unused
    ):
        # Ensure litellm can route the model — add provider prefix if needed
        self.model_name = _normalize_model_name(model_name)
        self.prompt_dir = prompt_dir
        self.coding_guidelines = coding_guidelines
        self.coding_system_prompt = coding_system_prompt
        self.adata_summary = adata_summary
        self.paper_summary = paper_summary
        self.logger = logger
        self.use_self_critique = use_self_critique
        self.use_documentation = use_documentation
        self.max_iterations = max_iterations
        self.deepresearch_background = deepresearch_background
        self.log_prompts = log_prompts

    def _complete_structured(self, messages: list) -> dict:
        """Call LiteLLM via instructor and return a validated AnalysisPlan dict."""
        is_ollama = self.model_name.startswith("ollama")

        if is_ollama:
            return self._complete_structured_ollama(messages)

        result = _instructor_client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            response_model=AnalysisPlan,
            max_retries=3,
        )
        return result.model_dump()

    # ------------------------------------------------------------------
    # Ollama-specific structured output with reinforced schema prompting
    # ------------------------------------------------------------------

    _OLLAMA_SCHEMA_HINT = (
        "\n\n--- REQUIRED OUTPUT FORMAT ---\n"
        "You MUST respond with ONLY a valid JSON object with these exact keys:\n"
        "{\n"
        '  "hypothesis": "<string: your scientific hypothesis>",\n'
        '  "analysis_plan": ["<string: step 1>", "<string: step 2>"],\n'
        '  "first_step_code": "<string: complete Python code for the first step>",\n'
        '  "code_description": "<string: 1-2 sentences describing the code>",\n'
        '  "summary": "<string: 1-2 sentence overall summary>"\n'
        "}\n"
        "ALL FIVE fields are REQUIRED: hypothesis, analysis_plan, first_step_code, "
        "code_description, summary.\n"
        "Do NOT include any text outside the JSON object."
    )

    def _complete_structured_ollama(self, messages: list) -> dict:
        """Structured output for Ollama models with schema reinforcement and fallback."""
        messages = list(messages)

        # Reinforce the expected JSON schema at the end of the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i] = {
                    **messages[i],
                    "content": messages[i]["content"] + self._OLLAMA_SCHEMA_HINT,
                }
                break

        # Try instructor JSON-mode first (gives the model 5 chances)
        try:
            result = _instructor_client_json.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_model=AnalysisPlan,
                max_retries=5,
            )
            return result.model_dump()
        except (ValidationError, Exception) as exc:
            # Instructor exhausted retries — attempt manual extraction from the
            # raw LLM response as a last resort.
            print(f"⚠️ Instructor structured output failed, attempting manual extraction: {exc}")

        raw_response = litellm.completion(model=self.model_name, messages=messages)
        raw_text = raw_response.choices[0].message.content or ""
        return self._extract_analysis_plan(raw_text)

    @staticmethod
    def _extract_analysis_plan(raw_text: str) -> dict:
        """Best-effort extraction of AnalysisPlan fields from raw LLM text."""
        import re as _re
        text = raw_text.strip()

        # Strip markdown code fences (with or without language tag)
        if "```" in text:
            fence_pat = _re.compile(r"```(?:json)?\s*\n?(.*?)```", _re.DOTALL)
            for m in fence_pat.finditer(text):
                candidate = m.group(1).strip()
                if candidate.startswith("{"):
                    text = candidate
                    break
            else:
                # Unclosed fence (truncated) — split fallback
                for block in text.split("```"):
                    candidate = block.strip()
                    if candidate.startswith("json"):
                        candidate = candidate[4:].strip()
                    if candidate.startswith("{"):
                        text = candidate
                        break

        # Try direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "hypothesis" in data:
                return AnalysisPlan(**data).model_dump()
        except (json.JSONDecodeError, ValidationError):
            pass

        # Brute-force: find the outermost { ... } in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict) and "hypothesis" in data:
                    return AnalysisPlan(**data).model_dump()
            except (json.JSONDecodeError, ValidationError):
                pass

        # Try repairing truncated JSON (model ran out of tokens)
        if start != -1:
            from cellvoyager.execution.legacy import _repair_truncated_json
            repaired = _repair_truncated_json(text[start:])
            if repaired is not None and "hypothesis" in repaired:
                try:
                    return AnalysisPlan(**repaired).model_dump()
                except ValidationError:
                    pass

        raise ValueError(
            "Could not extract a valid AnalysisPlan from the model response. "
            "The local model may be too small for this task — try a larger model "
            "like ollama/llama3.1 (8B) or ollama/qwen2.5-coder (7B).\n"
            f"Raw response (first 500 chars): {raw_text[:500]}"
        )

    def _complete(self, messages: list) -> str:
        """Call LiteLLM for plain-text responses (e.g. critique feedback)."""
        response = litellm.completion(model=self.model_name, messages=list(messages))
        return response.choices[0].message.content

    def generate_jupyter_summary(self, notebook_cells):
        """Generate a comprehensive summary of notebook cells including source code and outputs (including errors)"""
        if notebook_cells is None:
            return ""

        jupyter_summary = ""
        for cell in notebook_cells:
            if cell["cell_type"] == "code" or cell["cell_type"] == "markdown" or cell["cell_type"] == "error":
                jupyter_summary += f"{cell['source']}\n"

        return jupyter_summary

    def generate_initial_analysis(self, attempted_analyses):
        prompt = open(os.path.join(self.prompt_dir, "first_draft.txt")).read()
        prompt = prompt.format(
            CODING_GUIDELINES=self.coding_guidelines,
            adata_summary=self.adata_summary,
            past_analyses=attempted_analyses,
            paper_txt=self.paper_summary,
            deepresearch_background=self.deepresearch_background,
            max_iterations=self.max_iterations,
        )

        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "Initial Analysis")

        return self._complete_structured([
            {"role": "system", "content": self.coding_system_prompt},
            {"role": "user", "content": prompt},
        ])

    def critique_step(self, analysis, past_analyses, notebook_cells, num_steps_left):
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        first_step_code = analysis["first_step_code"]

        # Generate comprehensive jupyter summary including outputs and errors
        jupyter_summary = self.generate_jupyter_summary(notebook_cells)

        if self.use_documentation:
            prompt = open(os.path.join(self.prompt_dir, "critic.txt")).read()
            # Get relevant documentation on the single-cell packages being used in the first step code
            try:
                documentation = get_documentation(first_step_code)
            except Exception as e:
                print(f"⚠️ Documentation extraction failed: {e}")
                documentation = ""
            prompt = prompt.format(
                hypothesis=hypothesis,
                analysis_plan=analysis_plan,
                first_step_code=first_step_code,
                CODING_GUIDELINES=self.coding_guidelines,
                adata_summary=self.adata_summary,
                past_analyses=past_analyses,
                paper_txt=self.paper_summary,
                jupyter_notebook=jupyter_summary,
                documentation=documentation,
                num_steps_left=num_steps_left,
            )
        else:
            prompt = open(os.path.join(self.prompt_dir, "ablations", "critic_NO_DOCUMENTATION.txt")).read()
            prompt = prompt.format(
                hypothesis=hypothesis,
                analysis_plan=analysis_plan,
                first_step_code=first_step_code,
                CODING_GUIDELINES=self.coding_guidelines,
                adata_summary=self.adata_summary,
                past_analyses=past_analyses,
                paper_txt=self.paper_summary,
                jupyter_notebook=jupyter_summary,
                num_steps_left=num_steps_left,
            )

        return self._complete([
            {
                "role": "system",
                "content": "You are a single-cell bioinformatics expert providing feedback on code and analysis plan.",
            },
            {"role": "user", "content": prompt},
        ])

    def incorporate_critique(self, analysis, feedback, notebook_cells, num_steps_left):
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        first_step_code = analysis["first_step_code"]

        # Generate comprehensive jupyter summary including outputs and errors
        jupyter_summary = self.generate_jupyter_summary(notebook_cells)

        prompt = open(os.path.join(self.prompt_dir, "incorporate_critque.txt")).read()
        prompt = prompt.format(
            hypothesis=hypothesis,
            analysis_plan=analysis_plan,
            first_step_code=first_step_code,
            CODING_GUIDELINES=self.coding_guidelines,
            adata_summary=self.adata_summary,
            feedback=feedback,
            jupyter_notebook=jupyter_summary,
            num_steps_left=num_steps_left,
        )

        # For Ollama models: reinforce constraints so the critique doesn't hallucinate
        if self.model_name.startswith("ollama"):
            prompt += (
                "\n\nCRITICAL CONSTRAINTS:\n"
                "- Keep the hypothesis closely related to the ORIGINAL hypothesis above.\n"
                "- The data is in AnnData (.h5ad) format — load with sc.read_h5ad(). "
                "Do NOT use read_csv, read_cellranger, or any other data loading method.\n"
                "- Only use these packages: scanpy, scvi, anndata, matplotlib, numpy, "
                "seaborn, pandas, scipy. Do NOT import any other packages.\n"
                "- The variable `adata` is already loaded in the kernel.\n"
            )

        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "Incorporate Critiques")

        return self._complete_structured([
            {"role": "system", "content": self.coding_system_prompt},
            {"role": "user", "content": prompt},
        ])

    def get_feedback(self, analysis, past_analyses, notebook_cells, num_steps_left, iterations=1):
        current_analysis = analysis
        is_ollama = self.model_name.startswith("ollama")
        for i in range(iterations):
            try:
                feedback = self.critique_step(current_analysis, past_analyses, notebook_cells, num_steps_left)
                revised = self.incorporate_critique(
                    current_analysis, feedback, notebook_cells, num_steps_left
                )
                # For small local models, validate the revised analysis isn't degraded.
                # Common failure mode: model loses context and hallucinates packages/data paths.
                if is_ollama and not self._is_valid_revision(current_analysis, revised):
                    print("⚠️ Self-critique produced a degraded analysis — keeping original")
                    continue
                current_analysis = revised
            except Exception as e:
                print(f"⚠️ Self-critique iteration {i+1} failed: {e} — keeping current analysis")
                continue

        return current_analysis

    @staticmethod
    def _is_valid_revision(original: dict, revised: dict) -> bool:
        """Check whether a revised analysis is plausible (not hallucinated)."""
        code = revised.get("first_step_code", "")
        # Reject if the code imports packages not in the allowed set
        _ALLOWED_IMPORTS = {
            "scanpy", "sc", "scvi", "anndata", "ad", "matplotlib", "plt",
            "numpy", "np", "seaborn", "sns", "pandas", "pd", "scipy",
            "stats", "warnings", "os", "sys", "json", "re", "math",
            "collections", "itertools", "functools", "pathlib",
        }
        import re as _re
        imports = set()
        for m in _re.finditer(r"^\s*(?:import|from)\s+(\w+)", code, _re.MULTILINE):
            imports.add(m.group(1))
        hallucinated = imports - _ALLOWED_IMPORTS
        if hallucinated:
            print(f"  ⚠️ Revised code imports unknown packages: {hallucinated}")
            return False
        # Reject if the code tries to load data from files that aren't .h5ad
        # (common hallucination: read_csv, read_cellranger, etc.)
        if _re.search(r"read_csv|read_excel|read_cellranger|\.csv|\.tsv|\.rdR", code):
            if "read_h5ad" not in code and "adata" not in code.split("read_csv")[0] if "read_csv" in code else True:
                print("  ⚠️ Revised code tries to load non-h5ad data files")
                return False
        # Reject if the hypothesis became completely generic
        hyp = revised.get("hypothesis", "").lower()
        _GENERIC_MARKERS = ["environmental conditions", "generic analysis", "general exploration"]
        if any(g in hyp for g in _GENERIC_MARKERS):
            orig_hyp = original.get("hypothesis", "").lower()
            if not any(g in orig_hyp for g in _GENERIC_MARKERS):
                print("  ⚠️ Revised hypothesis became generic")
                return False
        return True

    def generate_idea(self, past_analyses, analysis_idx=None, seeded_hypothesis=None):
        """
        Phase 1: Idea Generation

        Args:
            past_analyses: String of past analysis summaries
            analysis_idx: Analysis index for logging (optional)
            seeded_hypothesis: Simple hypothesis string to guide AI generation (optional)

        Returns:
            dict: Analysis containing hypothesis, analysis_plan, first_step_code, etc.
        """
        if seeded_hypothesis is not None:
            print(f"🌱 Using seeded hypothesis: {seeded_hypothesis}")
            return self.generate_analysis_from_hypothesis(seeded_hypothesis, past_analyses, analysis_idx)

        print("🧠 Generating new analysis idea...")

        # Create the initial analysis plan
        analysis = self.generate_initial_analysis(past_analyses)

        if analysis_idx is not None:
            step_name = f"{analysis_idx+1}_1"
            hypothesis = analysis["hypothesis"]
            analysis_plan = analysis["analysis_plan"]
            initial_code = analysis["first_step_code"]

            # Log only the output of the analysis
            self.logger.log_response(
                f"Hypothesis: {hypothesis}\n\nAnalysis Plan:\n"
                + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)])
                + f"\n\nInitial Code:\n{initial_code}",
                f"initial_analysis_{step_name}",
            )

        # Get feedback for the initial analysis plan and modify it accordingly
        if self.use_self_critique:
            modified_analysis = self.get_feedback(analysis, past_analyses, None, self.max_iterations)

            if analysis_idx is not None:
                self.logger.log_response(
                    f"APPLIED INITIAL SELF-CRITIQUE - Analysis {analysis_idx+1}",
                    f"self_critique_{step_name}",
                )

                hypothesis = modified_analysis["hypothesis"]
                analysis_plan = modified_analysis["analysis_plan"]
                current_code = modified_analysis["first_step_code"]

                # Log revised analysis plan
                self.logger.log_response(
                    f"Revised Hypothesis: {hypothesis}\n\nRevised Analysis Plan:\n"
                    + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)])
                    + f"\n\nRevised Code:\n{current_code}",
                    f"revised_analysis_{step_name}",
                )

            return modified_analysis
        else:
            if analysis_idx is not None:
                print("🚫 Skipping feedback on next step (no self-critique)")
                self.logger.log_response(
                    f"SKIPPING INITIAL SELF-CRITIQUE - Analysis {analysis_idx+1}",
                    f"no_self_critique_{step_name}",
                )

            return analysis

    def generate_analysis_from_hypothesis(self, hypothesis, past_analyses, analysis_idx=None):
        """
        Generate an analysis plan from a simple hypothesis string using AI

        Args:
            hypothesis: Simple hypothesis string
            past_analyses: String of past analysis summaries
            analysis_idx: Analysis index for logging (optional)

        Returns:
            dict: Analysis containing hypothesis, analysis_plan, first_step_code, etc.
        """
        # Create a modified prompt that incorporates the seeded hypothesis
        prompt = open(os.path.join(self.prompt_dir, "ablations", "analysis_from_hypothesis.txt")).read()
        prompt = prompt.format(
            hypothesis=hypothesis,
            coding_guidelines=self.coding_guidelines,
            adata_summary=self.adata_summary,
            paper_summary=self.paper_summary,
        )

        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "Seeded Hypothesis Analysis")

        analysis = self._complete_structured([
            {"role": "system", "content": self.coding_system_prompt},
            {"role": "user", "content": prompt},
        ])

        analysis = self.get_feedback(analysis, past_analyses, None, self.max_iterations)

        # Ensure the hypothesis matches what was provided
        analysis["hypothesis"] = hypothesis

        # Log the seeded hypothesis analysis
        if analysis_idx is not None:
            step_name = f"{analysis_idx+1}_1"
            analysis_plan = analysis["analysis_plan"]
            initial_code = analysis["first_step_code"]

            # Log the seeded hypothesis analysis
            self.logger.log_response(
                f"Seeded Hypothesis: {hypothesis}\n\nGenerated Analysis Plan:\n"
                + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)])
                + f"\n\nInitial Code:\n{initial_code}",
                f"seeded_hypothesis_{step_name}",
            )

        return analysis
