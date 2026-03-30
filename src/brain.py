import json
import asyncio
from datetime import datetime, timezone
from groq import AsyncGroq
import ollama
import time
import re
from google import genai
from google.genai import types

# Local module imports
from config import (
    ANALYZE_SPECIFIC_PROMPT, 
    DETECT_SYMBOL_PROMPT, 
    GENERATE_SEARCH_QUERY_PROMPT, 
    GET_COIN_PROFILE_PROMPT,
    LLM_CONFIG,
    ANALYZE_GENERAL_PROMPT
)
from utils import search_web_sync, coin_categories

# Constants
RATE_LIMIT_BUFFER = 0.2
MAX_LLM_RETRIES = 3
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_LLM_MAX_TOKENS = 1024

class AgentBrain:
    def __init__(self, use_groqcloud=True, api_key=None, groqcloud_model="google/gemini-2.0-flash-exp:free", use_gemini = False, google_api_key = None, gemini_model = "gemma-3-27b-it"):
        self.use_groqcloud = use_groqcloud
        self.model = groqcloud_model
        self.ollama_model = "nexus-qwen3"  # Fallback
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
                ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': 'hi'}], keep_alive=-1, options={'num_ctx': 2048})
                print("[SYSTEM] Model loaded!")
            except Exception as e:
                print(f"[WARNING] Model load issue: {e}")

    async def _wait_for_rate_limit(self):
        """
        Rate limit wait for GroqCloud/OpenRouter.
        """
        if not self.use_groqcloud:
            return

        current_time = time.time()
        time_diff = current_time - self.last_request_time
        self.last_request_time = time.time()

    def _extract_json(self, text):
        """
        Removes LLM conversational filler and extracts only the JSON block.
        """
        if not text:
            return ""
        
        try:
            # 1. Clean markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            # 2. Extract JSON by finding outermost curly braces
            start = text.find('{')
            end = text.rfind('}')
            
            if start != -1 and end != -1:
                return text[start:end+1]
            
            return text.strip()
        except Exception:
            return text.strip()

    # TODO: consider splitting
    async def _submit_to_llm(self, prompt, temperature=0.1, json_mode=True, max_tokens=1024, use_system_prompt=True, reasoning_mode="none", compound_custom=None):
        """
        Central LLM Call Function
        """
        retries = 0
        while retries < MAX_LLM_RETRIES:
            try:
                messages_payload = []
                
                messages_payload.append({"role": "system", "content": LLM_CONFIG['system_prompt']})
                
                messages_payload.append({"role": "user", "content": prompt})

                # --- A. OPENROUTER / GROQ ---
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
                # --- B. GOOGLE GEMINI / OLLAMA ---
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
                else:
                    options = {
                        'temperature': temperature
                    }
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
                    
                    # --- 429 RATE LIMIT EXTRACTION LOGIC ---
                    if "429" in error_msg:
                        retries += 1
                        
                        # Identify wait time from error message (ms or s)
                        ms_match = re.search(r"try again in (\d+)ms", error_msg)
                        sec_match = re.search(r"try again in (\d+)s", error_msg)
                        
                        wait_time = 1.0 
                        
                        if ms_match:
                            wait_time = float(ms_match.group(1)) / 1000.0
                        elif sec_match:
                            wait_time = float(sec_match.group(1))
                        
                        # Add safety margin
                        wait_time += RATE_LIMIT_BUFFER
                        
                        print(f"[RATE LIMIT] 429 Error! Waiting {wait_time:.2f}s... (Attempt {retries}/{MAX_LLM_RETRIES})")
                        await asyncio.sleep(wait_time)
                        continue 
                    
                    else:
                        print(f"[ERROR] LLM Request Failed: {e}")
                        return None

    async def analyze_specific(self, news, symbol, price, changes, search_context="", coin_full_name="Unknown", market_cap_str="", rsi_val=0, btc_trend=0, volume_24h="", funding_rate=0):
        # 1. Profile Info
        await self._wait_for_rate_limit()
        coin_category = await self.get_coin_profile(symbol)
        current_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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
        Identifies the relevant crypto symbol from news text using LLM.
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
        Generates an optimized web search query based on news and symbol.
        """
        prompt = GENERATE_SEARCH_QUERY_PROMPT.format(
            news=news,
            symbol=symbol.upper()
        )
        
        response_text = await self._submit_to_llm(prompt, temperature=0.7, json_mode=False, max_tokens=64, use_system_prompt=False, reasoning_mode="none")
        return response_text.strip()

    async def get_coin_profile(self, symbol):
        sym = symbol.upper().replace('USDT', '')
        
        # 1. FAST LIST
        if sym in coin_categories:
            return coin_categories[sym]

        # 2. CACHE CHECK
        if sym in self.coin_cache:
            return self.coin_cache[sym]

        # 3. INTERNET SEARCH & LLM
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
        Analyzes news using technical context only, bypassing web research.
        """
        await self._wait_for_rate_limit()

        # Populate prompt without research context
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