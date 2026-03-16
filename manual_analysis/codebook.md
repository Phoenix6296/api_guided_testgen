# Codebook for Document Classification

## Overview

This codebook provides the systematic taxonomy and coding framework developed through our thematic analysis process. It serves as the authoritative guide for classifying documents based on their influence on test generation.

## Coding Framework

### Document Classification Categories

#### 1. Bug-inducing
**Definition**: Documents that reveal actual software defects, incorrect behavior, or unintended functionality that should be fixed

**Inclusion Criteria**:
- Reports of software defects or incorrect behavior
- API behavior that contradicts documentation or expected functionality
- Crashes, failures, or incorrect results that are NOT intentional
- Reproduction steps for problematic behavior that represents a bug
- Issues labeled as "bug", "defect", or "regression"
- Unexpected exceptions that indicate software problems

**Exclusion Criteria**:
- Expected exceptions that are part of normal API operation
- Proper error handling demonstrations (even if they show exceptions)
- User errors or misuse of APIs
- Performance issues that are not bugs
- Expected validation errors or input checking

**Examples**:
- "API returns wrong result when input is X"
- "Function crashes with segmentation fault"
- "Memory leak in specific usage pattern"
- "API behaves differently than documented"

**Key Distinguisher**: The document reveals something is BROKEN or WRONG in the software itself

#### 2. Standard Use Cases
**Definition**: Documents showing typical, expected API usage patterns and basic functionality

**Inclusion Criteria**:
- Official API documentation examples
- Basic, straightforward usage patterns
- Standard parameter combinations
- Expected behavior descriptions
- Comprehensive parameter documentation

**Exclusion Criteria**:
- Complex or unusual parameter combinations
- Edge cases or boundary conditions
- Performance optimization examples
- Chains of multiple API sequences

**Examples**:
- Basic API usage from official documentation
- Standard parameter value examples
- Typical workflow demonstrations
- Common use case illustrations

#### 3. API Usage Sequences
**Definition**: Documents demonstrating proper API call sequences, workflows, and multi-step operations

**Inclusion Criteria**:
- Multi-step API workflows
- Sequential API calls with dependencies
- Pipeline or workflow examples
- Integration between multiple APIs

**Exclusion Criteria**:
- Single API call examples
- Independent API operations
- Simple parameter examples

**Examples**:
- Model training workflows
- Data preprocessing pipelines
- Multi-step transformations
- API integration patterns

#### 4. Unique/Complex Input/Parameters Handling
**Definition**: Documents with sophisticated, edge-case, or complex parameter or input usage

**Inclusion Criteria**:
- Complex parameter combinations
- Edge case parameter values
- Boundary condition examples
- Advanced parameter usage patterns

**Exclusion Criteria**:
- Basic parameter examples
- Standard parameter values
- Simple usage patterns

**Examples**:
- Advanced tensor shapes and configurations
- Complex data type specifications
- Edge case parameter combinations
- Sophisticated input configurations

#### 5. Exception Handling
**Definition**: Documents showing proper error handling, exception management, and robustness patterns for EXPECTED error conditions

**Inclusion Criteria**:
- Proper exception handling patterns (try-catch, error checking)
- Input validation and error prevention strategies
- Graceful handling of expected failure conditions
- Error recovery and cleanup procedures
- Documentation of expected exceptions and their handling
- Best practices for robust error management

**Exclusion Criteria**:
- Bug reports revealing actual software defects
- Unexpected crashes or failures (those are bug-inducing)
- Performance issues without error handling focus
- Standard operation examples without error considerations

**Examples**:
- "How to handle FileNotFoundError when loading data"
- "Proper exception handling in API calls"
- "Input validation to prevent errors"
- "Graceful degradation when network fails"

**Key Distinguisher**: The document shows how to PROPERLY HANDLE expected errors and exceptions, not reports of broken functionality

#### 6. Integration Patterns
**Definition**: Documents showing how APIs work together in larger systems or with external libraries

**Inclusion Criteria**:
- Integration with other libraries
- System-level examples
- Inter-API interactions
- External system connections

**Exclusion Criteria**:
- Single library examples
- Isolated API usage
- Independent operations

**Examples**:
- Integration with NumPy/Pandas
- Cross-library compatibility
- System integration examples
- External tool integration

#### 7. Performance Considerations
**Definition**: Documents addressing performance optimization, efficiency, and best practices

**Inclusion Criteria**:
- Performance optimization examples
- Efficiency considerations
- Best practice recommendations
- Resource usage patterns

**Exclusion Criteria**:
- Basic usage examples
- Functional examples without performance focus
- Standard operation patterns

**Examples**:
- Memory optimization strategies
- Computational efficiency examples
- Performance best practices
- Resource management patterns

## Coding Instructions

### Step 1: Initial Assessment
1. Read the document content carefully
2. Identify the primary purpose and focus
3. Note key characteristics and patterns
4. Consider the test generation impact

### Step 2: Category Selection
1. Review all category definitions
2. Select the most appropriate primary category
3. If multiple categories apply, choose the most dominant one
4. Document rationale for selection

### Step 3: Supporting Evidence
1. Identify specific text or code that supports the classification
2. Note key phrases or patterns
3. Document the connection to test generation impact
4. Record any ambiguities or edge cases

### Step 4: Quality Checks
1. Verify consistency with previous classifications
2. Check for potential disagreements
3. Document any uncertainties
4. Prepare for consensus discussion if needed

## Inter-rater Reliability

### Agreement Criteria
- **High Agreement**: Same category selected by both coders
- **Moderate Agreement**: Related categories selected (e.g., bug_inducing vs exception_handling)
- **Low Agreement**: Unrelated categories selected

### Disagreement Resolution
1. Review original document together
2. Discuss rationale for each classification
3. Consult codebook definitions
4. Seek consensus through discussion
5. Document final decision and rationale

## Quality Assurance

### Consistency Checks
- Regular review of classifications
- Cross-validation between coders
- Periodic codebook updates
- Documentation of edge cases

### Validation Process
- Inter-rater reliability calculation
- Codebook refinement based on feedback
- Final validation with full dataset

## References

This codebook was developed following established qualitative research methodologies for thematic analysis and systematic taxonomy development, drawing from:
- Guest, G., MacQueen, K. M., & Namey, E. E. (2011). Applied thematic analysis
- Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology
- Saldaña, J. (2015). The coding manual for qualitative researchers