from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from base import (
    Sponsor, BusinessRequirement, BusinessValue, SpecialIssue, SystemRequest
)
import json

class PromptConfig(BaseModel):
    """Configuration for prompt behavior"""
    temperature: float = 0.3
    max_tokens: int = 2000
    strict_json: bool = True
    max_retries: int = 2
    agent_role: str = "Business Analyst"
    importance: str = "high"

class PromptResult(BaseModel):
    """Result container for prompt building"""
    messages: List[BaseMessage]
    model_kwargs: Dict[str, Any]
    debug_info: Dict[str, Any]

class PromptFactory:
    """Central factory for all workflow prompts"""
    
    def __init__(self):
        self.templates = self._init_templates()
        self.schema_descriptions = self._init_schemas()
    
    def build_prompt(
        self, 
        node_name: str,
        input_data: str,
        context: Dict[str, Any] = None,
        config: PromptConfig = None,
        target_schema: Type[BaseModel] = None
    ) -> PromptResult:
        """Main entry point for prompt construction"""
        
        config = config or PromptConfig()
        context = context or {}
        
        # Get node template
        template = self.templates.get(node_name)
        if not template:
            raise ValueError(f"No template found for node: {node_name}")
        
        # Build system message
        system_content = self._build_system_message(
            template, config, target_schema, context
        )
        
        # Build user message
        user_content = self._build_user_message(
            template, input_data, context
        )
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content)
        ]
        
        # Add few-shot examples if available
        if template.get("examples"):
            example_msgs = self._build_examples(template["examples"])
            messages.extend(example_msgs)
        
        model_kwargs = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }
        
        debug_info = {
            "node_name": node_name,
            "agent_role": config.agent_role,
            "target_schema": target_schema.name if target_schema else None,
            "context_keys": list(context.keys())
        }
        
        return PromptResult(
            messages=messages,
            model_kwargs=model_kwargs,
            debug_info=debug_info
        )
    
    def build_retry_prompt(
        self, 
        original_output: str, 
        errors: List[str],
        target_schema: Type[BaseModel]
    ) -> List[BaseMessage]:
        """Build prompt for retry after parsing failure"""
        
        schema_desc = self.schema_descriptions.get(target_schema.__name__, "")
        
        fix_content = f"""The JSON you returned failed validation:

            ORIGINAL OUTPUT:
            {original_output}

            VALIDATION ERRORS:
            {chr(10).join(f'- {error}' for error in errors)}

            REQUIRED SCHEMA:
            {schema_desc}

            Return corrected JSON only, no explanations."""
        
        return [HumanMessage(content=fix_content)]
    
    def _build_system_message(
        self, 
        template: Dict, 
        config: PromptConfig,
        target_schema: Optional[Type[BaseModel]],
        context: Dict[str, Any]
    ) -> str:
        """Build system message from template"""
        
        # Base system header
        system_parts = [
            self._get_system_header(config.agent_role),
            template.get("instructions", ""),
        ]
        
        # Add context if provided
        if context:
            context_block = self._build_context_block(context)
            system_parts.append(context_block)
        
        # Add output contract for structured output
        if target_schema and config.strict_json:
            schema_desc = self.schema_descriptions.get(target_schema.name)
            if schema_desc:
                output_contract = self._build_output_contract(schema_desc)
                system_parts.append(output_contract)
        
        return "\n\n".join(filter(None, system_parts))
    
    def _build_user_message(
        self, 
        template: Dict, 
        input_data: str, 
        context: Dict[str, Any]
    ) -> str:
        """Build user message with input data"""
        
        user_template = template.get("user_template", "{input}")
        
        # Simple template substitution
        return user_template.format(
            input=input_data,
            **context
        )
    
    def _build_context_block(self, context: Dict[str, Any]) -> str:
        """Build context information block"""
        
        context_items = []
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                context_items.append(f"{key}: {len(value) if isinstance(value, list) else 'provided'}")
            else:
                context_items.append(f"{key}: {str(value)[:100]}")
        
        return f"CONTEXT:\n{chr(10).join(f'- {item}' for item in context_items)}"
    
    def _build_output_contract(self, schema_description: str) -> str:
        """Build output format contract"""
        
        return f"""OUTPUT CONTRACT:
            Return only valid JSON matching this schema:

            {schema_description}

            Rules:
            - Return JSON only, no explanations
            - Use null for unknown values
            - Do not add extra fields
            - Ensure all required fields are present"""
    
    def _build_examples(self, examples: List[Dict]) -> List[BaseMessage]:
        """Build few-shot examples"""
        
        messages = []
        for example in examples:
            if "input" in example and "output" in example:
                messages.extend([
                    HumanMessage(content=example["input"]),
                    HumanMessage(content=example["output"])  # Using HumanMessage for simplicity
                ])
        return messages
    
    def _get_system_header(self, agent_role: str) -> str:
        """Get system header based on agent role"""
        
        headers = {
            "Business Analyst": "You are a Business Analyst specialized in requirements engineering and system analysis. Be precise, factual, and ask clarifying questions when uncertain.",
            "Requirements Engineer": "You are a Requirements Engineer focused on extracting functional and technical requirements. Ensure completeness and avoid assumptions.",
            "Stakeholder Analyst": "You are a Stakeholder Analyst expert at identifying key project sponsors and their concerns. Focus on roles and organizational impact."
        }
        
        return headers.get(agent_role, headers["Business Analyst"])
    
    def _init_templates(self) -> Dict[str, Dict]:
        """Initialize node-specific templates"""
        
        return {
            "problem_definition": {
                "instructions": """Extract and clarify the core business problem from the input. 
                
                Your task:
                1. Identify the main business problem or opportunity
                2. Note any ambiguous areas that need clarification
                3. Provide a concise problem statement

                If the problem is unclear, ask ONE specific clarifying question.""",
                                
                                "user_template": "Analyze this business situation and define the core problem:\n\n{input}",
                                
                                "examples": [
                                    {
                                        "input": "Our customer service is slow and customers are complaining",
                                        "output": "Core problem: Customer service response times exceed customer expectations, leading to dissatisfaction. Clarifying question: What is the current average response time and what would be considered acceptable?"
                                    }
                                ]
                            },
                            
            "business_need": {
                "instructions": """Transform the problem definition into a clear business need statement.

                Your task:
                1. Create a concise business need (1-2 sentences)
                2. Focus on what the organization needs to achieve
                3. Avoid technical solutions - focus on business outcomes""",
                                
                "user_template": "Based on this problem definition, articulate the business need:\n\n{input}"
                            },
                            
            "sponsor_determination": {
                "instructions": """Identify key project sponsors who would champion this initiative.

                Your task:
                1. Identify 2-4 potential sponsors
                2. Consider organizational hierarchy and business impact
                3. Include name (can be role-based) and specific role
                4. Focus on decision-makers and budget owners""",
                                    
                "user_template": "Given this business need, identify the key sponsors who would support this project:\n\n{input}",
                
                "examples": [
                    {
                        "input": "Need to improve customer service response times",
                        "output": '{"sponsors": [{"name": "VP of Customer Experience", "role": "Executive Sponsor - owns customer satisfaction metrics"}, {"name": "Customer Service Director", "role": "Operational Sponsor - manages daily operations"}]}'
                    }
                ]
                            },
            
            "requirements_determination": {
                "instructions": """Extract business requirements from the sponsor's perspective.

                Your task:
                1. Identify 3-5 high-level capabilities the system needs
                2. Focus on what the system must DO, not how
                3. Consider this specific sponsor's role and concerns
                4. Make requirements testable and specific""",
                                
                "user_template": "From the perspective of {sponsor_name} ({sponsor_role}), what business requirements are needed for:\n\n{business_need}"
                            },
                            
            "value_determination": {
                "instructions": """Identify business values from the sponsor's perspective.

                Your task:
                1. Identify 2-4 tangible and intangible benefits
                2. Focus on outcomes this sponsor cares about
                3. Consider both quantifiable and strategic benefits
                4. Connect benefits to sponsor's role and responsibilities""",
                                
                "user_template": "From the perspective of {sponsor_name} ({sponsor_role}), what business values would result from:\n\n{business_need}"
                            },
                            
            "specialissues": {
                "instructions": """Identify constraints and special considerations from the sponsor's perspective.

                Your task:
                1. Identify 1-3 constraints or special considerations
                2. Consider budget, timeline, regulatory, or technical constraints
                3. Focus on issues this sponsor would be concerned about
                4. Include both hard constraints and risk factors""",
                                
                "user_template": "From the perspective of {sponsor_name} ({sponsor_role}), what constraints or special issues must be considered for:\n\n{business_need}"
                            }
                        }
                    
    def _init_schemas(self) -> Dict[str, str]:
        """Initialize schema descriptions for structured output"""
        
        return {
            "Need": """
                    {
                        "need": "string - Detailed business need description (required)"
                    }""",
                                
            "Sponsors": """
                    {
                    "sponsors": [
                            {
                            "name": "string - Name or role of sponsor (required)",
                            "role": "string - Specific organizational role and responsibility (required)"
                            }
                    ]
                    }""",
                    
            "Requirements": """
                    {
                    "requirements": [
                            {
                            "capability": "string - High-level business capability description (required)"
                            }
                    ]
                    }""",
                    
            "Values": """
                    {
                    "values": [
                            {
                            "value": "string - Tangible or intangible business benefit (required)"
                            }
                    ]
                    }""",
                    
            "Constraints": """
                    {
                    "constraints": [
                            {
                            "issue": "string - Constraint or critical consideration (required)"
                            }
                    ]
                    }"""
        }

# Validation utilities
def validate_and_retry(
    model_output: str, 
    target_schema: Type[BaseModel], 
    factory: PromptFactory,
    max_retries: int = 2
) -> tuple[BaseModel, List[str]]:
    """Validate model output and retry on failure"""
    
    errors = []
    
    for attempt in range(max_retries + 1):
        try:
            # Try to parse as JSON first
            json_data = json.loads(model_output)
            # Then validate with Pydantic
            return target_schema(**json_data), []
            
        except json.JSONDecodeError as e:
            errors.append(f"JSON parsing error: {str(e)}")
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        # If not last attempt, generate retry prompt
        if attempt < max_retries:
            # In practice, you'd call the model again with retry prompt
            # retry_messages = factory.build_retry_prompt(model_output, errors, target_schema)
            pass
    
    return None, errors

# Factory singleton
prompt_factory = PromptFactory()

# Convenience functions for each node
def build_problem_definition_prompt(input_text: str, config: PromptConfig = None) -> PromptResult:
    return prompt_factory.build_prompt("problem_definition", input_text, config=config)

def build_business_need_prompt(problem_def: str, config: PromptConfig = None) -> PromptResult:
    from base import Need  # Assuming Need is defined
    return prompt_factory.build_prompt(
        "business_need", 
        problem_def, 
        config=config,
        target_schema=Need
    )

def build_sponsor_prompt(business_need: str, config: PromptConfig = None) -> PromptResult:
    from base import Sponsors  # Assuming Sponsors wrapper exists
    return prompt_factory.build_prompt(
        "sponsor_determination", 
        business_need,
        config=config, 
        target_schema=Sponsors
    )

def build_requirements_prompt(sponsor: Sponsor, business_need: str, config: PromptConfig = None) -> PromptResult:
    from base import Requirements  # Assuming Requirements wrapper exists
    context = {
        "sponsor_name": sponsor.name,
        "sponsor_role": sponsor.role,
        "business_need": business_need
    }
    return prompt_factory.build_prompt(
        "requirements_determination",
        business_need,
        context=context,
        config=config,
        target_schema=Requirements
    )

def build_values_prompt(sponsor: Sponsor, business_need: str, config: PromptConfig = None) -> PromptResult:
    from base import Values  # Assuming Values wrapper exists
    context = {
        "sponsor_name": sponsor.name,
        "sponsor_role": sponsor.role,
        "business_need": business_need
    }
    return prompt_factory.build_prompt(
        "value_determination",
        business_need,
        context=context,
        config=config,
        target_schema=Values
    )

def build_constraints_prompt(sponsor: Sponsor, business_need: str, config: PromptConfig = None) -> PromptResult:
    from base import Constraints  # Assuming Constraints wrapper exists
    context = {
        "sponsor_name": sponsor.name,
        "sponsor_role": sponsor.role,
        "business_need": business_need
    }
    return prompt_factory.build_prompt(
        "specialissues",
        business_need,
        context=context,
        config=config,
        target_schema=Constraints
    )

Sponsors = Sponsor(name="Alex",role="Assistance")


result = prompt_factory.build_prompt(
    "sponsor_determination", 
    "create a website",
    target_schema=Sponsors
)

print(result)