"""
Safety Validator
RUBRIC: Governance & Guardrails - Safety validator with Azure Content Safety 

TASK: Implement safety validation using both local checks and Azure Content Safety
"""
import re
from typing import Dict, Any, List
from src.config import Config

class SafetyValidator:
    """Validates content safety and detects adversarial attacks (jailbreaks)"""

    def __init__(self):
        from guardrails.content_safety import ContentSafety  # lazy
        self.content_safety = ContentSafety()
        
        # HINT: Define prompt injection patterns
        self.injection_patterns = [
            r"ignore previous instructions",  # HINT: ["ignore previous instructions", "bypass safety", "override guidelines", "you are now in developer mode", "delete all data", etc.]
            r"bypass safety",
            r"override guidelines",
            r"you are now in developer mode",
            r"delete all data", 
             r"act as (an? )?(unrestricted|unfiltered|jailbreak)",
        ]
        
        self.client = None
        if Config.AZURE_CONTENT_SAFETY_ENDPOINT and Config.AZURE_CONTENT_SAFETY_KEY:
            try:
                from azure.ai.contentsafety import ContentSafetyClient      # lazy
                from azure.core.credentials import AzureKeyCredential       # lazy
                self.client = ContentSafetyClient(
                    endpoint=Config.AZURE_CONTENT_SAFETY_ENDPOINT,
                    credential=AzureKeyCredential(Config.AZURE_CONTENT_SAFETY_KEY)
                )
            except Exception as e:
                print(f"Warning: Failed to init Azure Content Safety: {e}")

    def validate(self, text: str, severity_threshold: str = "high") -> Dict[str, Any]:
        """
        Validates the text for safety violations
        
        HINT: This method should:
        1. Initialize flags list and is_safe boolean
        2. Run local content safety check
        3. Check for prompt injection patterns
        4. Run Azure Content Safety check if client available
        5. Return dict with is_safe, flags, severity
        """
        flags = []
        is_safe = True
        
        # HINT: 1. Local Content Safety Guardrail (Keywords & Regex)
        local_result = self.content_safety.check(text)
        if not local_result['is_safe']:
            is_safe = False  # HINT: False
            for flag in local_result['flags']:  # HINT: 'flags'
                flags.append(f"Unsafe Keyword ({flag['category']}): {flag['keyword']}")

        # HINT: 2. Specific Injection Checks
        for pattern in self.injection_patterns: 
            if re.search(pattern, text, re.IGNORECASE): 
                is_safe = False  
                flags.append(f"Prompt Injection Detected: {pattern}")
        
        # HINT: 3. Azure Content Safety Check
        if self.client:
            try:
                from azure.ai.contentsafety.models import AnalyzeTextOptions  # lazy
                from azure.core.exceptions import HttpResponseError            # lazy
                request = AnalyzeTextOptions(text=text)
                response = self.client.analyze_text(request)
                if response.categories_analysis:
                    for analysis in response.categories_analysis:
                        if analysis.severity > 2:
                            is_safe = False
                            flags.append(f"Azure Content Safety Violation: {analysis.category} ({analysis.severity})")
            except Exception as e:
                print(f"Azure Content Safety check failed: {e}")
        
        return {
            'is_safe': is_safe,  
            'flags': flags,   
            'severity': "high" if not is_safe else "low" 
        }