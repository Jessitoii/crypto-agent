"""
AI Reasoning Engine (Brain)

This module implements the core intelligence layer of the Crypto-Agent.
It handles LLM orchestration across multiple providers (Gemini, GroqCloud/OpenRouter, Ollama),
news sentiment analysis, symbol detection, and sector classification.

The engine uses a tiered priority system for model selection and includes
robust error handling for rate limits and JSON extraction.
"""

import json
import asyncio
from datetime import datetime, timezone
from groq import AsyncGroq
import ollama
import time
import re
from google import genai
from google.genai import types

# --- Local module imports ---
from config import (
    ANALYZE_SPECIFIC_PROMPT, 
    DETECT_SYMBOL_PROMPT, 
    GENERATE_SEARCH_QUERY_PROMPT, 
    GET_COIN_PROFILE_PROMPT,
    LLM_CONFIG,
    ANALYZE_GENERAL_PROMPT
)
from utils import search_web_sync, coin_categories

# --- Constants ---
RATE_LIMIT_BUFFER = 0.2
MAX_LLM_RETRIES = 3
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_LLM_MAX_TOKENS = 1024

class AgentBrain:
    """
    Orchestrates AI-driven analysis and decision making.
    
    This class manages connections to various LLM backends and provides
    high-level methods for processing market news and technical data.
    
    Attributes:
        use_groqcloud (bool): Flag to use GroqCloud/OpenRouter.
        model (str): The primary model identifier for OpenRouter.
        ollama_model (str): Fallback local model identifier.
        api_key (str): Authentication key for the primary LLM provider.
        coin_cache (dict): In-memory cache for coin sector classifications.
        last_request_time (float): Timestamp of the last outbound LLM request.
    """
    def __init__(self, use_groqcloud=True, api_key=None, groqcloud_model="google/gemini-2.0-flash-exp:free", use_gemini = False, google_api_key = None, gemini_model = "gemma-3-27b-it"):
        """
        Initializes the AgentBrain with specified provider configurations.
        
        Args:
            use_groqcloud (bool): Enable OpenRouter/Groq access.
            api_key (str, optional): API key for OpenRouter.
            groqcloud_model (str): Model name for OpenRouter.
            use_gemini (bool): Enable Google Gemini access.
            google_api_key (str, optional): API key for Gemini.
            gemini_model (str): Model name for Gemini.
        """
        self.use_groqcloud = use_groqcloud
        self.model = groqcloud_model
        self.ollama_model = "nexus-qwen3"  # Default local fallback
        self.api_key = api_key
        self.coin_cache = {}
        self.last_request_time = 0
        self.use_gemini = use_gemini
        self.google_api_key = google_api_key
        self.gemini_model = gemini_model

        # Initialize LLM Client based on priority: Gemini -> GroqCloud -> Ollama
        if self.use_gemini:
            print(f"[BRAIN] Mode: GOOGLE GEMINI ({self.gemini_model})")
            self.client = genai.Client(api_key=self.google_api_key)
        elif self.use_groqcloud:
            print(f"[BRAIN] Mode: OPENROUTER ({self.model})")
            self.client = AsyncGroq(api_key=self.api_key)
        else:
            print(f"[BRAIN] Mode: LOCAL OLLAMA ({self.ollama_model})")
            print("[SYSTEM] Loading Model to VRAM (Keep-Alive)...")
            try:
                # Warm up local model to minimize first-request latency
                ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': 'hi'}], keep_alive=-1, options={'num_ctx': 2048})
                print("[SYSTEM] Model loaded!")
            except Exception as e:
                print(f"[WARNING] Model load issue: {e}")

    async def _wait_for_rate_limit(self):
        """
        Asynchronous rate limit management for remote LLM providers.
        
        Ensures a minimal buffer between requests to prevent 429 errors.
        """
        if not self.use_groqcloud:
            return

        current_time = time.time()
        time_diff = current_time - self.last_request_time
        # Logic to sleep if time_diff < required_delay would go here if needed
        self.last_request_time = time.time()

    def _extract_json(self, text):
        """
        Parses and extracts a clean JSON string from raw LLM output.
        
        Args:
            text (str): The raw string response from the LLM.
            
        Returns:
            str: The extracted JSON block or the original text if no block found.
        """
        if not text:
            return ""
        
        try:
            # 1. Strip markdown formatting if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            # 2. Heuristic extraction of the primary JSON object
            start = text.find('{')
            end = text.rfind('}')
            
            if start != -1 and end != -1:
                return text[start:end+1]
            
            return text.strip()
        except Exception:
            return text.strip()

    async def _submit_to_llm(self, prompt, temperature=0.1, json_mode=True, max_tokens=1024, use_system_prompt=True, reasoning_mode="none", compound_custom=None):
        """
        Centralized dispatcher for all outbound LLM requests.
        
        Args:
            prompt (str): The user prompt to send.
            temperature (float): Sampling temperature for creativity.
            json_mode (bool): If True, requests JSON-formatted response.
            max_tokens (int): Maximum length of the generated response.
            use_system_prompt (bool): If True, prepends the global system prompt.
            reasoning_mode (str): Execution mode for reasoning models ('none', 'default').
            compound_custom (dict, optional): Provider-specific extra parameters.
            
        Returns:
            str: The processed (and potentially JSON-cleaned) response text.
        """
        retries = 0
        while retries < MAX_LLM_RETRIES:
            try:
                messages_payload = []
                
                # Prepend system context if required for the task
                if use_system_prompt:
                    messages_payload.append({"role": "system", "content": LLM_CONFIG['system_prompt']})
                
                messages_payload.append({"role": "user", "content": prompt})

                # --- A. OPENROUTER / GROQ PIPELINE ---
                if self.use_groqcloud:
                    if compound_custom:
                        completion = await self.client.chat.completions.create(
                            model=self.gemini_model,
                            messages=messages_payload,
                            response_format={"type": "json_object"} if json_mode else None,
                            temperature=temperature,
                            compound_custom = compound_custom
                        )
                    else:
                        completion = await self.client.chat.completions.create(
                            model=self.model,
                            messages=messages_payload,
                            response_format={"type": "json_object"} if json_mode else None,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            reasoning_effort=reasoning_mode
                        )
                    raw_response = completion.choices[0].message.content
                    cleaned_response = self._extract_json(raw_response)
                    return cleaned_response

                # --- B. GOOGLE GEMINI PIPELINE ---
                elif self.use_gemini:
                    res = self.client.models.generate_content(
                        model=self.gemini_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature = temperature,
                        ),
                    )
                    if json_mode:
                        cleaned_response = self._extract_json(res.text)
                        return cleaned_response
                    else:
                        return res.text

                # --- C. LOCAL OLLAMA PIPELINE ---
                else:
                    options = {'temperature': temperature}
                    res = await asyncio.to_thread(
                        ollama.chat,
                        model=self.ollama_model,
                        messages=[{"role": "user", "content": prompt}],
                        format='json' if json_mode else '',
                        options=options,
                        keep_alive=-1
                    )
                    return res['message']['content']

            except Exception as e:
                error_msg = str(e)
                
                # Handle Rate Limiting (HTTP 429) dynamically based on retry headers
                if "429" in error_msg:
                    retries += 1
                    
                    ms_match = re.search(r"try again in (\d+)ms", error_msg)
                    sec_match = re.search(r"try again in (\d+)s", error_msg)
                    
                    wait_time = 1.0 
                    if ms_match:
                        wait_time = float(ms_match.group(1)) / 1000.0
                    elif sec_match:
                        wait_time = float(sec_match.group(1))
                    
                    wait_time += RATE_LIMIT_BUFFER
                    print(f"[RATE LIMIT] 429 Error! Waiting {wait_time:.2f}s... (Attempt {retries}/{MAX_LLM_RETRIES})")
                    await asyncio.sleep(wait_time)
                    continue 
                else:
                    print(f"[ERROR] LLM Request Failed: {e}")
                    return None

    async def analyze_specific(self, news, symbol, price, changes, search_context="", coin_full_name="Unknown", market_cap_str="", rsi_val=0, btc_trend=0, volume_24h="", funding_rate=0):
        """
        Performs detailed qualitative and quantitative analysis on a specific coin.
        
        Args:
            news (str): Current news item.
            symbol (str): Asset ticker (e.g., BTC).
            price (float): Current asset price.
            changes (dict): Price delta history.
            search_context (str): Supplemental information from web research.
            coin_full_name (str): Canonical name of the asset.
            market_cap_str (str): Formatted market capitalization.
            rsi_val (float): Relative Strength Index value.
            btc_trend (float): General BTC market trend correlation.
            volume_24h (str): Formatted 24-hour trading volume.
            funding_rate (float): Perpetual futures funding rate.
            
        Returns:
            dict: Trade recommendation including 'action', 'confidence', and 'reason'.
        """
        await self._wait_for_rate_limit()
        coin_category = await self.get_coin_profile(symbol)
        current_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Construct the specialized prompt for deep analysis
        prompt = ANALYZE_SPECIFIC_PROMPT.format(
            symbol=symbol.upper(),
            coin_full_name=coin_full_name,
            market_cap_str=market_cap_str,
            coin_category=coin_category,
            rsi_val=rsi_val,
            btc_trend=btc_trend,
            volume_24h=volume_24h,
            funding_rate=funding_rate,
            current_time_str=current_time_str,
            price=price,
            change_1m=changes['1m'],
            change_10m=changes['10m'],
            change_1h=changes['1h'],
            change_24h=changes['24h'],
            news=news,
            search_context=search_context
        )

        response_text = await self._submit_to_llm(prompt, temperature=0.1, json_mode=True, max_tokens=2048, use_system_prompt=True, reasoning_mode="default")
        
        try:
            return json.loads(response_text)
        except Exception:
            return {"action": "HOLD", "confidence": 0, "reason": "Error parsing JSON"}

    async def detect_symbol(self, news, available_pairs):
        """
        Identifies relevant crypto symbols within a text string.
        
        Args:
            news (str): News text to analyze.
            available_pairs (list): List of tradable asset pairs.
            
        Returns:
            str: The identified symbol (e.g., "BTC") or None if not found.
        """
        prompt = DETECT_SYMBOL_PROMPT.format(news=news)
        compound_custom = {
            "tools":{
                "enabled_tools":["web_search","code_interpreter","visit_website"]
            }
        }
        response_text = await self._submit_to_llm(prompt, temperature=0.0, json_mode=True, use_system_prompt=False, compound_custom=compound_custom)
        
        try:
            res_json = json.loads(response_text)
            return res_json.get('symbol')
        except Exception as e:
            print(f"[ERROR] Symbol Detect JSON error: {e}")
            return None

    async def generate_search_query(self, news, symbol):
        """
        Synthesizes an optimized web search query from raw news and ticker.
        
        Args:
            news (str): The raw news input.
            symbol (str): The asset ticker.
            
        Returns:
            str: A refined search query string.
        """
        prompt = GENERATE_SEARCH_QUERY_PROMPT.format(
            news=news,
            symbol=symbol.upper()
        )
        
        response_text = await self._submit_to_llm(prompt, temperature=0.7, json_mode=False, max_tokens=64, use_system_prompt=False, reasoning_mode="none")
        return response_text.strip()

    async def get_coin_profile(self, symbol):
        """
        Retrieves the sector classification/utility profile of a coin.
        
        Utilizes a hierarchical search: Hardcoded -> Local Cache -> LLM Research.
        
        Args:
            symbol (str): Ticker to research.
            
        Returns:
            str: Sector classification (e.g., "DeFi", "Layer 1").
        """
        sym = symbol.upper().replace('USDT', '')
        
        # Level 1: Static Lookup
        if sym in coin_categories:
            return coin_categories[sym]

        # Level 2: Runtime Cache
        if sym in self.coin_cache:
            return self.coin_cache[sym]

        # Level 3: Dynamic Research via Web Search + LLM Synthesis
        print(f"[BRAIN] {sym} unknown, researching...")
        query = f"what is {sym} crypto category sector utility"
        
        try:
            search_text = await asyncio.to_thread(search_web_sync, query)
            
            profile_prompt = GET_COIN_PROFILE_PROMPT.format(
                search_text=search_text,
                symbol=sym
            )
            
            category = await self._submit_to_llm(profile_prompt, temperature=0.0, json_mode=False, max_tokens=256, use_system_prompt=False)
            category = category.strip()
            
            self.coin_cache[sym] = category
            print(f"[PROFILE] {symbol} classified: {category}")
            return category

        except Exception as e:
            print(f"Profile Error: {e}")
            return "Unknown"

    async def analyze_specific_no_research(self, news, symbol):
        """
        Conducts rapid sentiment analysis without external web research.
        
        Useful for high-frequency news ingestion where latency is critical.
        
        Args:
            news (str): News text.
            symbol (str): Ticker.
            
        Returns:
            dict: Trade recommendation object.
        """
        await self._wait_for_rate_limit()

        prompt = ANALYZE_GENERAL_PROMPT.format(
            symbol=symbol.upper(),
            news=news,
        )

        compound_custom = {
            "tools":{
                "enabled_tools":["web_search","code_interpreter","visit_website"]
            }
        }
        response_text = await self._submit_to_llm(prompt, temperature=0.1, json_mode=True, max_tokens=1024, compound_custom = compound_custom)
        self.last_request_time = time.time()
        
        try:
            return json.loads(response_text)
        except Exception:
            return {"action": "HOLD", "confidence": 0, "reason": "Error parsing simulation JSON"}