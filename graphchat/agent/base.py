from typing import List, Optional, Union, Literal, Annotated, List, Dict, Set, Any, Tuple
from pydantic import BaseModel, Field,  model_validator
import uuid
"""
System Request BaseModels
"""
class Sponsor(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str = Field(..., description="Name of the person who will serve as the primary contact for the project.")
    role: str = Field(..., description="The sponsor's role in the organization (e.g., 'Director of Marketing').")

class BusinessRequirement(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    capability: str = Field(..., description="A high-level business capability the system will need to have.")

class BusinessValue(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    value: str = Field(..., description="A tangible or intangible benefit the organization expects from the system.")

class SpecialIssue(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    issue: str = Field(..., description="A constraint or other critical information to be considered (e.g., deadlines, budget limits).")

class SystemRequest(BaseModel):
    """
    Models the formal System Request document, which initiates a project.
    It captures the core business drivers and justification for a new system.
    """
    project_name: str = Field(..., description="The official name of the proposed project.")
    
    # REFINEMENT 2: Business need is a central, required field.
    business_need: str = Field(..., description="The core business problem or opportunity driving this request.")
    
    # REFINEMENT 1: Using direct lists instead of collection classes.
    # The default_factory ensures an empty list is created if none is provided.
    sponsors: List[Sponsor] = Field(
        default_factory=list, 
        description="The key stakeholders who are sponsoring and advocating for the project."
    )
    
    business_requirements: List[BusinessRequirement] = Field(
        default_factory=list,
        description="The high-level capabilities the system must have to address the business need."
    )
    
    business_values: List[BusinessValue] = Field(
        default_factory=list,
        description="The expected benefits to the organization, both tangible and intangible."
    )
    
    special_issues: List[SpecialIssue] = Field(
        default_factory=list,
        description="Any constraints, deadlines, or other special considerations."
    )

"""
Feasibility Analysis Models
"""

class FeasibilityFinding(BaseModel):
    """A generic finding for technical or organizational feasibility."""
    id: int
    finding: str = Field(..., description="A specific finding or assessment point.")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Assessed risk for this specific point.")

class EconomicFinding(BaseModel):
    """A specific financial finding for the economic feasibility analysis."""
    id: int
    type: Literal["Development Cost", "Operational Cost", "Tangible Benefit", "Intangible Benefit"]
    description: str
    estimated_value: float = Field(..., description="Monetary value. Costs should be negative, benefits positive.")
    timing: Literal["One-Time", "Annual"] = Field("One-Time", description="Is the cost/benefit recurring?")

class FeasibilityAnalysis(BaseModel):
    """
    Models the formal Feasibility Analysis report, which assesses the viability
    and risks of the project proposed in the System Request.
    """
    system_request: SystemRequest 
    
    executive_summary: str = Field(..., description="A one-paragraph summary of the key findings and recommendation.")
    overall_risk_assessment: Literal["Low", "Medium", "High"] = Field(..., description="A holistic assessment of the project's combined risk.")
    
    # Technical and Organizational sections use the generic finding model.
    technical_feasibility: List[FeasibilityFinding] = Field(default_factory=list, description="Assessment of 'Can we build it?'")
    organizational_feasibility: List[FeasibilityFinding] = Field(default_factory=list, description="Assessment of 'If we build it, will they use it?'")
    
    # The Economic section now uses the more detailed, quantitative model.
    economic_feasibility: List[EconomicFinding] = Field(default_factory=list, description="Assessment of 'Should we build it?'")
    
    # REFINEMENT 1: The recommendation is now a precise, controlled value.
    recommendation: Literal[
        "Proceed", 
        "Proceed with Conditions", 
        "Revise and Resubmit", 
        "Do Not Proceed"
    ] = Field(..., description="The final, official recommendation for the project.")

"""
Analysis Modeling

Requirement Defination
a document that lists 
the new systems capabilities. It then describes how to analyze requirements using requirements
analysis strategies and how to gather requirements using interviews, JAD sessions, 
quest
"""


class FunctionalRequirement(BaseModel):
    """
    Models a statement of what the system must do or what information it must contain.
    Typically relates to a process or data.
    """
    id: str = Field(..., description="A unique identifier for the requirement (e.g., 'FR01').")
    description: str = Field(..., description="A detailed, testable description of the function.")
    priority: Literal["High", "Medium", "Low"] = Field("High", description="The priority for implementing this requirement.")

class NonFunctionalRequirement(BaseModel):
    """
    Models a behavioral property that the system must have, such as performance or usability.
    """
    id: str = Field(..., description="A unique identifier for the requirement (e.g., 'NFR-PERF-01').")
    
    # REFINEMENT 1: Renamed 'type' to 'category' and added a 'name' field for a summary.
    category: Literal["Operational", "Performance", "Security", "Cultural and Political"] = Field(
        ..., description="The main category of the non-functional requirement."
    )
    name: str = Field(..., description="A short, descriptive name for the requirement (e.g., 'Data Encryption at Rest').")
    description: str = Field(..., description="The detailed, measurable, and testable definition of the requirement.")

class RequirementDefinition(BaseModel):  
    """
    Models the formal Requirements Definition document, which serves as a checklist
    of all functional and non-functional requirements for the system.
    """
    # REFINEMENT 3: Added project context.
    project_name: str
    version: str = "1.0"
    
    requirements: List[Union[FunctionalRequirement, NonFunctionalRequirement]] = Field(default_factory=list)

    # REFINEMENT 2: Added validator to ensure all requirement IDs are unique.
    @model_validator(mode='after')
    def check_unique_requirement_ids(self):
        seen_ids: Set[str] = set()
        for req in self.requirements:
            if req.id in seen_ids:
                raise ValueError(f"Duplicate requirement ID '{req.id}' found. IDs must be unique.")
            seen_ids.add(req.id)
        return self
"""
Use Case Diagram
"""

class Actor(BaseModel):
    id: int = Field(..., description="Unique ID for the actor within its diagram.")
    name: str
    description: str = Field(..., description="A brief description of this actor's role.")

class UseCaseDetails(BaseModel):
    goal: str = Field(..., description="The high-level goal this use case achieves for the primary actor.")

class UseCase(BaseModel):
    id: int = Field(..., description="Unique ID for the use case within its diagram.")
    name: str
    details: UseCaseDetails = Field(..., description="The core functional description of the use case.")

class Relationship(BaseModel):
    relationship_type: Literal["association", "include", "extend", "generalization"]
    source_id: int
    source_type: Literal["actor", "use_case"]
    target_id: int
    target_type: Literal["use_case", "actor"]

class UseCaseReference(BaseModel):
    diagram_id: int
    use_case_id: int

class CrossDiagramRelationship(BaseModel):
    relationship_type: Literal["include", "extend"]
    source: UseCaseReference
    target: UseCaseReference
    description: str = Field(..., description="A brief justification for this cross-diagram link.")
  
class UseCaseDiagram(BaseModel):
    id: int
    name: str
    actors: List[Actor] = Field(default_factory=list)
    use_cases: List[UseCase] = Field(default_factory=list, max_length=15)
    relationships: List[Relationship] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_diagram_integrity(self):
        # 1. Check for unique IDs within the diagram
        actor_ids = {actor.id for actor in self.actors}
        if len(actor_ids) != len(self.actors):
            raise ValueError(f"Diagram ID {self.id}: Duplicate actor IDs found.")
        
        use_case_ids = {uc.id for uc in self.use_cases}
        if len(use_case_ids) != len(self.use_cases):
            raise ValueError(f"Diagram ID {self.id}: Duplicate use case IDs found.")

        # 2. Validate internal relationships
        for rel in self.relationships:
            if rel.source_type == 'actor' and rel.source_id not in actor_ids:
                raise ValueError(f"Diagram {self.id}: Relationship references non-existent source actor ID {rel.source_id}.")
            if rel.source_type == 'use_case' and rel.source_id not in use_case_ids:
                raise ValueError(f"Diagram {self.id}: Relationship references non-existent source use case ID {rel.source_id}.")
            if rel.target_type == 'actor' and rel.target_id not in actor_ids:
                raise ValueError(f"Diagram {self.id}: Relationship references non-existent target actor ID {rel.target_id}.")
            if rel.target_type == 'use_case' and rel.target_id not in use_case_ids:
                raise ValueError(f"Diagram {self.id}: Relationship references non-existent target use case ID {rel.target_id}.")
        
        return self

class UseCaseModelCollection(BaseModel):
    project_name: str = Field(description="An appropriate name for this project.")
    diagrams: List[UseCaseDiagram] = Field(default_factory=list)
    cross_diagram_relationships: List[CrossDiagramRelationship] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_collection_integrity(self):
        # 1. Check for unique diagram IDs
        diagram_ids = {diag.id for diag in self.diagrams}
        if len(diagram_ids) != len(self.diagrams):
            raise ValueError("Duplicate diagram IDs found in the collection.")

        # 2. Build a master map of all use cases for cross-reference validation
        valid_refs: Dict[int, Set[int]] = {
            diag.id: {uc.id for uc in diag.use_cases} for diag in self.diagrams
        }

        # 3. Validate cross-diagram relationships
        for i, cross_rel in enumerate(self.cross_diagram_relationships):
            src, tgt = cross_rel.source, cross_rel.target
            
            if src.diagram_id not in valid_refs:
                raise ValueError(f"Cross-rel #{i}: Source diagram ID {src.diagram_id} does not exist.")
            if src.use_case_id not in valid_refs.get(src.diagram_id, set()):
                raise ValueError(f"Cross-rel #{i}: Source use case ID {src.use_case_id} not found in diagram {src.diagram_id}.")
            
            if tgt.diagram_id not in valid_refs:
                raise ValueError(f"Cross-rel #{i}: Target diagram ID {tgt.diagram_id} does not exist.")
            if tgt.use_case_id not in valid_refs.get(tgt.diagram_id, set()):
                raise ValueError(f"Cross-rel #{i}: Target use case ID {tgt.use_case_id} not found in diagram {tgt.diagram_id}.")
                
        return self

"""
Activity Diagram
    Action
    Activity
    object Node
    control flow

"""


# --- Activity Elements ---
class ActivityNode(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    node_type: Literal["start", "end", "action", "decision", "merge", "fork", "join"]
    
    # REFINEMENT 1: Explicitly link a node to a swimlane.
    swimlane_id: Optional[int] = Field(None, description="The ID of the swimlane this node belongs to.")
    
class ActivityFlow(BaseModel):
    source_id: uuid.UUID 
    target_id: uuid.UUID 
    guard_condition: Optional[str] = Field(None, description="A condition that must be true for this flow to be taken (e.g., '[Order Approved]').")

class Swimlane(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    # This actor_id must exist in the linked Use Case Diagram
    actor_id: Optional[uuid.UUID] = Field(None, description="The ID of the Actor from the linked use case diagram responsible for this lane.")

class UseCaseLink(BaseModel):
    diagram_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    use_case_id: uuid.UUID = Field(default_factory=uuid.uuid4)

class ActivityDiagram(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    use_case_links: List[UseCaseLink] = Field(..., min_length=1, description="Links this workflow to at least one use case.")
    swimlanes: List[Swimlane] = Field(default_factory=list)
    nodes: List[ActivityNode] = Field(default_factory=list, max_length=30)
    flows: List[ActivityFlow] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_diagram_integrity(self):
        # REFINEMENT 2: Comprehensive internal validation.
        node_ids: Set[uuid.UUID] = {node.id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError(f"Activity Diagram {self.id}: Duplicate node IDs found.")
        
        swimlane_ids: Set[uuid.UUID] = {sl.id for sl in self.swimlanes}
        if len(swimlane_ids) != len(self.swimlanes):
            raise ValueError(f"Activity Diagram {self.id}: Duplicate swimlane IDs found.")
        
        # Validate flows
        for flow in self.flows:
            if flow.source_id not in node_ids:
                raise ValueError(f"Activity Diagram {self.id}: Flow references non-existent source node ID {flow.source_id}.")
            if flow.target_id not in node_ids:
                raise ValueError(f"Activity Diagram {self.id}: Flow references non-existent target node ID {flow.target_id}.")

        # Validate node-to-swimlane assignments
        for node in self.nodes:
            if node.swimlane_id is not None and node.swimlane_id not in swimlane_ids:
                raise ValueError(f"Activity Diagram {self.id}: Node {node.id} is assigned to non-existent swimlane ID {node.swimlane_id}.")
        
        return self

class ActivityModelCollection(BaseModel):
    project_name: str
    use_case_collection: UseCaseModelCollection  # The source of truth for use cases and actors
    activity_diagrams: List[ActivityDiagram] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_cross_model_references(self):
        # REFINEMENT 3: Cross-model validation.
        uc_coll = self.use_case_collection
        
        # Build master lookup maps from the use case collection
        valid_use_cases: Set[tuple[int, int]] = set()
        valid_actors: Dict[int, Set[int]] = {} # {diagram_id: {actor_id_1, actor_id_2}}
        for uc_diag in uc_coll.diagrams:
            valid_actors[uc_diag.id] = {actor.id for actor in uc_diag.actors}
            for uc in uc_diag.use_cases:
                valid_use_cases.add((uc_diag.id, uc.id))

        for ad in self.activity_diagrams:
            # Validate UseCaseLinks
            for uc_link in ad.use_case_links:
                if (uc_link.diagram_id, uc_link.use_case_id) not in valid_use_cases:
                    raise ValueError(
                        f"In Activity Diagram {ad.id}, link to non-existent Use Case (Diagram ID {uc_link.diagram_id}, "
                        f"UC ID {uc_link.use_case_id}) is invalid."
                    )
            
            # Validate Swimlane actor references
            for sl in ad.swimlanes:
                if sl.actor_id is not None:
                    # An activity diagram can be linked to multiple use cases from different diagrams.
                    # We must check if the actor exists in ANY of the linked use case diagrams.
                    actor_is_valid = False
                    for uc_link in ad.use_case_links:
                        if sl.actor_id in valid_actors.get(uc_link.diagram_id, set()):
                            actor_is_valid = True
                            break
                    if not actor_is_valid:
                        raise ValueError(
                            f"In Activity Diagram {ad.id}, Swimlane '{sl.name}' references non-existent Actor ID {sl.actor_id} "
                            f"in any of its linked use case diagrams."
                        )
        return self


"""
Use case Description
"""

class Precondition(BaseModel):
    id: int
    condition: str

class Postcondition(BaseModel):
    id: int
    condition: str

class Step(BaseModel):
    id: int
    actor: str
    action: str
    system_response: Optional[str] = None

class Extension(BaseModel):
    step_number: int = Field(..., description="The ID of the step in the main flow where this extension can occur.")
    condition: str = Field(..., description="The condition that triggers this alternative flow.")
    steps: List[Step]
    return_to_step: Optional[int] = Field(None, description="The ID of the main flow step to resume at, if any.")

class UseCaseDescription(BaseModel):
    use_case_id: int
    diagram_id: int
    
    primary_actor: str
    goal: str
    scope: Literal["system", "business", "white-box", "black-box"] = "system"
    level: Literal["summary", "user-goal", "subfunction"] = "user-goal"
    
    preconditions: List[Precondition] = Field(default_factory=list)
    success_guarantee: List[Postcondition] = Field(default_factory=list)
    minimal_guarantee: List[Postcondition] = Field(default_factory=list)
    
    main_success_scenario: List[Step] = Field(..., min_length=1)
    extensions: List[Extension] = Field(default_factory=list)
    
    frequency: Optional[str] = None
    priority: Literal["high", "medium", "low"] = "medium"

    # REFINEMENT 1: Added validator for internal logical integrity.
    @model_validator(mode='after')
    def validate_flow_integrity(self):
        # 1. Check for unique step IDs in the main success scenario.
        main_step_ids: Set[int] = {s.id for s in self.main_success_scenario}
        if len(main_step_ids) != len(self.main_success_scenario):
            raise ValueError(f"UC ({self.diagram_id},{self.use_case_id}): Duplicate step IDs found in the main success scenario.")
        
        # 2. Validate that extensions refer to existing steps.
        for i, ext in enumerate(self.extensions):
            if ext.step_number not in main_step_ids:
                raise ValueError(f"UC ({self.diagram_id},{self.use_case_id}): Extension #{i} references non-existent main step ID {ext.step_number}.")
            if ext.return_to_step is not None and ext.return_to_step not in main_step_ids:
                raise ValueError(f"UC ({self.diagram_id},{self.use_case_id}): Extension #{i} has an invalid return_to_step ID {ext.return_to_step}.")
        
        return self

class UseCaseDescriptionCollection(BaseModel):
    project_name: str
    descriptions: List[UseCaseDescription] = Field(default_factory=list)

    # REFINEMENT 1: Added validator for collection-level integrity.
    @model_validator(mode='after')
    def validate_description_uniqueness(self):
        seen_keys: Set[tuple[int, int]] = set()
        for desc in self.descriptions:
            key = (desc.diagram_id, desc.use_case_id)
            if key in seen_keys:
                raise ValueError(f"Duplicate use case description found for diagram_id={desc.diagram_id}, use_case_id={desc.use_case_id}.")
            seen_keys.add(key)
        return self


"""
Structure Modeling
"""
"""
Object Identification
"""
class CandidateObject(BaseModel):
    """Represents a single potential object/class discovered during analysis."""
    id: int
    name: str = Field(..., description="The name of the candidate object (typically a noun).")
    
    discovery_method: Literal["textual_analysis", "brainstorming", "common_object_list", "pattern"] = Field(
        ..., description="The technique that first identified this candidate."
    )
    
    category: Literal["physical", "role", "event", "interaction", "abstract"] = Field(
        ..., description="Classification based on common object lists."
    )
    
    status: Literal["candidate", "accepted", "rejected", "merged"] = Field(
        default="candidate", description="The current state of this candidate in the analysis process."
    )
    
    justification: Optional[str] = Field(
        None, description="Reasoning for the current status (e.g., 'Rejected: Out of scope', 'Merged into User class')."
    )
    
    merged_into_id: Optional[int] = Field(
        None, description="If status is 'merged', this points to the ID of the object it was merged into."
    )

class ObjectIdentificationSession(BaseModel):
    """
    Represents the entire object identification process for a given problem domain.
    It holds the source text and the single, evolving list of candidate objects.
    """
    domain_description: str = Field(
        ..., description="The source text (e.g., use case) used for textual analysis."
    )
    
    candidate_objects: List[CandidateObject] = Field(
        default_factory=list, 
        description="A single, unified list of all candidate objects discovered and analyzed."
    )

    def get_finalized_classes(self) -> List[CandidateObject]:
        """A helper method to easily retrieve the final list of accepted classes."""
        return [obj for obj in self.candidate_objects if obj.status == "accepted"]

    """
    Why This Model is Better
        Single Source of Truth: There is only one list, candidate_objects. An object's entire history is contained within its own properties.
        Reflects the Process: The status field (candidate -> accepted/rejected/merged) perfectly models the filtering lifecycle.
        Enhanced Traceability: The justification field provides critical project memory, explaining why decisions were made. The merged_into_id field explicitly tracks refinement.
        Completeness: It includes brainstorming as a discovery method and uses the book's terminology.
        Practicality: The get_finalized_classes() helper method makes it easy to get the final result without cluttering the core data model.

    """

"""
Class-Responsibility-colaborator
"""

class Responsibility(BaseModel):
    """Represents a single responsibility and the collaborators needed to fulfill it."""
    description: str = Field(..., description="A short verb phrase describing what the class does or knows.")
    
    collaborators: List[str] = Field(
        default_factory=list, 
        description="List of other class names this class must interact with for this specific responsibility."
    )

class CRCCard(BaseModel):
    """Models a single Class-Responsibility-Collaboration card."""
    class_name: str = Field(..., description="The name of the class (singular noun).")
    description: str = Field(..., description="A brief sentence describing the purpose of the class.")
    
    # Inheritance relationships
    superclass: Optional[str] = Field(None, description="The name of the parent class, if any.")
    subclasses: List[str] = Field(default_factory=list, description="List of child classes, if any.")
    
    # Responsibilities are split into "knowing" (attributes) and "doing" (methods)
    attributes: List[str] = Field(
        default_factory=list, 
        description="The 'knowing' responsibilities, which will become attributes."
    )
    
    responsibilities: List[Responsibility] = Field(
        default_factory=list,
        description="The 'doing' responsibilities, which will become methods."
    )

class CRCModel(BaseModel):
    """
    Represents the entire collection of CRC cards for a system or subsystem.
    This class is responsible for ensuring the integrity of the entire model.
    """
    project_name: str
    cards: List[CRCCard]
    
    @model_validator(mode='after')
    def validate_collaborations_and_inheritance(self):
        """
        Ensures that every collaborator and super/subclass mentioned on a card
        corresponds to another valid card in the model. This is critical for
        maintaining the integrity of the object model.
        """
        # Create a set of all valid class names for fast lookups
        valid_class_names: Set[str] = {card.class_name for card in self.cards}
        
        for card in self.cards:
            # Validate superclass
            if card.superclass and card.superclass not in valid_class_names:
                raise ValueError(f"On card '{card.class_name}', superclass '{card.superclass}' does not exist in the model.")
            
            # Validate subclasses
            for subclass in card.subclasses:
                if subclass not in valid_class_names:
                    raise ValueError(f"On card '{card.class_name}', subclass '{subclass}' does not exist in the model.")
            
            # Validate collaborators within each responsibility
            for resp in card.responsibilities:
                for collaborator in resp.collaborators:
                    if collaborator not in valid_class_names:
                        raise ValueError(
                            f"Validation Error on card '{card.class_name}': "
                            f"Responsibility '{resp.description}' has an invalid collaborator '{collaborator}'. "
                            f"'{collaborator}' is not a defined class in this model."
                        )
        return self

"""
Class Diagram
"""   
Visibility = Literal["public", "private", "protected"]

class Attribute(BaseModel):
    """Models an attribute of a class."""
    name: str
    type: str = Field("string", description="The data type of the attribute (e.g., string, int, bool, or another class name).")
    visibility: Visibility = "private"

class Parameter(BaseModel):
    """Models a single parameter for a method."""
    name: str
    type: str

class Method(BaseModel):
    """Models a method (operation) of a class."""
    name: str
    parameters: List[Parameter] = Field(default_factory=list)
    return_type: str = Field("void", description="The data type returned by the method.")
    visibility: Visibility = "public"

class UMLClass(BaseModel):
    """Models a single class in the diagram."""
    name: str = Field(..., description="The unique name of the class.")
    description: str = Field(..., description="A brief description of the class's purpose.")
    is_abstract: bool = False
    attributes: List[Attribute] = Field(default_factory=list)
    methods: List[Method] = Field(default_factory=list)

class Generalization(BaseModel):
    """Models an inheritance relationship (is-a)."""
    subclass: str = Field(..., description="The name of the class that inherits.")
    superclass: str = Field(..., description="The name of the class being inherited from.")

class Association(BaseModel):
    """Models an association, aggregation, or composition relationship."""
    type: Literal["association", "aggregation", "composition"] = "association"
    
    # Defines the two ends of the relationship
    class1_name: str
    class1_multiplicity: str = Field("1", description="e.g., '1', '0..1', '0..*', '*'")
    
    class2_name: str
    class2_multiplicity: str = Field("1", description="e.g., '1', '0..1', '0..*', '*'")
    
    description: Optional[str] = Field(None, description="An optional description of the relationship's meaning.")

class ClassDiagram(BaseModel):
    """
    Represents a complete Class Diagram, containing all classes and their relationships.
    This model enforces the structural integrity of the diagram.
    """
    project_name: str
    classes: List[UMLClass]
    associations: List[Association] = Field(default_factory=list)
    generalizations: List[Generalization] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_diagram_integrity(self):
        """
        Ensures the entire diagram is logically consistent:
        1. Class names are unique.
        2. All relationships refer to existing classes.
        3. Generalization (inheritance) is not circular.
        """
        # 1. Check for unique class names
        class_names: Set[str] = {cls.name for cls in self.classes}
        if len(class_names) != len(self.classes):
            raise ValueError("Duplicate class names found in the diagram.")
            
        # 2. Validate all relationships
        for assoc in self.associations:
            if assoc.class1_name not in class_names:
                raise ValueError(f"Association error: Class '{assoc.class1_name}' not found in the diagram.")
            if assoc.class2_name not in class_names:
                raise ValueError(f"Association error: Class '{assoc.class2_name}' not found in the diagram.")
        
        for gen in self.generalizations:
            if gen.subclass not in class_names:
                raise ValueError(f"Generalization error: Subclass '{gen.subclass}' not found in the diagram.")
            if gen.superclass not in class_names:
                raise ValueError(f"Generalization error: Superclass '{gen.superclass}' not found in the diagram.")
        
        # 3. Check for circular inheritance (e.g., A inherits B, B inherits A)
        # We build a dependency graph and check for cycles.
        inheritance_graph: Dict[str, str] = {gen.subclass: gen.superclass for gen in self.generalizations}
        for start_node in inheritance_graph:
            path = {start_node}
            curr_node = start_node
            while curr_node in inheritance_graph:
                curr_node = inheritance_graph[curr_node]
                if curr_node in path:
                    raise ValueError(f"Circular inheritance detected involving class '{curr_node}'.")
                path.add(curr_node)
        
        return self



"""
Object Diagram
"""
class Attribute(BaseModel):
    name: str

class UMLClass(BaseModel):
    name: str
    attributes: List[Attribute] = Field(default_factory=list)

class Association(BaseModel):
    class1_name: str
    class2_name: str

class ClassDiagram(BaseModel):
    classes: List[UMLClass]
    associations: List[Association] = Field(default_factory=list)


class AttributeValue(BaseModel):
    """Represents a specific value for an attribute of an object."""
    name: str = Field(..., description="The name of the attribute, must match the class definition.")
    value: Any = Field(..., description="The concrete value of the attribute at a moment in time.")

class DiagramObject(BaseModel):
    """Models a single object (an instance of a class)."""
    instance_name: str = Field(..., description="The unique name for this instance in the diagram (e.g., 'prof_hawking').")
    class_name: str = Field(..., description="The name of the class this object is an instance of (e.g., 'Professor').")
    attribute_values: List[AttributeValue] = Field(
        default_factory=list,
        description="The state of the object, showing specific values for its attributes."
    )

class Link(BaseModel):
    """Models a link (an instance of an association) between two objects."""
    object1_instance_name: str = Field(..., description="The instance name of the first object in the link.")
    object2_instance_name: str = Field(..., description="The instance name of the second object in the link.")
    association_description: Optional[str] = Field(None, description="Optional text describing the link's purpose.")

class ObjectDiagram(BaseModel):
    """
    Represents a complete Object Diagram. It is a snapshot of object instances
    that MUST be consistent with a corresponding Class Diagram.
    """
    name: str = Field(..., description="A name for this specific scenario or snapshot (e.g., 'Successful Course Registration').")
    
    # The source of truth for validation
    class_diagram: ClassDiagram
    
    # The instances and links in this snapshot
    objects: List[DiagramObject]
    links: List[Link] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_against_class_diagram(self):
        """
        This is the most critical validator. It ensures that this Object Diagram
        is a legal and valid instantiation of the provided Class Diagram.
        """
        class_diagram = self.class_diagram
        
        # Step 1: Build efficient lookups from the Class Diagram
        class_map: Dict[str, UMLClass] = {cls.name: cls for cls in class_diagram.classes}
        valid_associations: Set[Tuple[str, str]] = {
            tuple(sorted((asc.class1_name, asc.class2_name))) for asc in class_diagram.associations
        }

        # Step 2: Validate the objects
        object_map: Dict[str, DiagramObject] = {}
        for obj in self.objects:
            if obj.instance_name in object_map:
                raise ValueError(f"Duplicate instance name '{obj.instance_name}' found.")
            object_map[obj.instance_name] = obj

            if obj.class_name not in class_map:
                raise ValueError(f"Object '{obj.instance_name}' is instance of non-existent class '{obj.class_name}'.")
            
            uml_class = class_map[obj.class_name]
            valid_attribute_names: Set[str] = {attr.name for attr in uml_class.attributes}
            for attr_val in obj.attribute_values:
                if attr_val.name not in valid_attribute_names:
                    raise ValueError(f"Object '{obj.instance_name}' has invalid attribute '{attr_val.name}'. Valid attributes for '{obj.class_name}' are: {list(valid_attribute_names)}.")

        # Step 3: Validate the links
        for link in self.links:
            if link.object1_instance_name not in object_map:
                raise ValueError(f"Link error: Object '{link.object1_instance_name}' not found.")
            if link.object2_instance_name not in object_map:
                raise ValueError(f"Link error: Object '{link.object2_instance_name}' not found.")

            obj1 = object_map[link.object1_instance_name]
            obj2 = object_map[link.object2_instance_name]
            association_key = tuple(sorted((obj1.class_name, obj2.class_name)))
            
            if association_key not in valid_associations:
                raise ValueError(f"Invalid link between '{obj1.instance_name}:{obj1.class_name}' and '{obj2.instance_name}:{obj2.class_name}'. No association exists between these classes in the Class Diagram.")
        
        return self


"""
UML Model
"""


class FunctionalModel(BaseModel):
    """Aggregates all artifacts related to the functional view of the system."""
    use_case_diagrams: UseCaseModelCollection
    use_case_descriptions: UseCaseDescriptionCollection
    activity_diagrams: ActivityModelCollection

class StructuralModel(BaseModel):
    """Aggregates all artifacts related to the structural (data) view."""
    class_diagram: ClassDiagram
    object_diagrams: list[ObjectDiagram] = Field(default_factory=list, description="Example snapshots to validate the class diagram.")


class SystemProposal(BaseModel):
    """
    The master deliverable for the Analysis Phase. This document combines all
    analysis artifacts into a single, consistent, and complete package.
    """
    project_name: str
    version: str = "1.0"
    
    # Initiation Artifacts (provide the history and justification)
    system_request: SystemRequest
    feasibility_analysis: FeasibilityAnalysis
    
    # The core analysis models, organized by view
    requirements_definition: RequirementDefinition
    functional_model: FunctionalModel
    structural_model: StructuralModel
    

    @model_validator(mode='after')
    def validate_internal_consistency(self):
        """
        Ensures the entire proposal is internally consistent. This is the
        final, critical check of the analysis phase.
        """
        # 1. Check for consistent project names across all artifacts
        project_names = {
            self.project_name,
            self.system_request.project_name,
            # Add checks for other sub-models if they were to retain project_name
        }
        if len(project_names) > 1:
            raise ValueError(f"Inconsistent project names found across proposal documents: {project_names}")

        # 2. Verify that models that depend on each other are using the same source of truth.
        # Check that the Activity Diagrams are based on the correct Use Case model.
        if self.functional_model.activity_diagrams.use_case_collection is not self.functional_model.use_case_diagrams:
             raise ValueError("ActivityModelCollection must reference the same UseCaseModelCollection instance provided in the FunctionalModel.")
        
        # 3. Add other critical consistency checks here. For example:
        # - Ensure every UseCase in the diagram model has a corresponding UseCaseDescription.
        # - Ensure every Class in the ClassDiagram was derived from the CRC cards/Object ID process.
        # - Ensure the structural and behavioral models are consistent with the functional model.
        
        print("✅ System Proposal internal consistency checks passed.")
        return self
