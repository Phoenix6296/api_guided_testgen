# 🔧 Prompt Engineering Design

This directory contains the prompt engineering evolution for automated unit test generation across ML/DL libraries.

## 🔄 Evolution Process

Our prompt engineering followed an iterative refinement process to reach the final prompts used in the research:

### Version 1: Initial Basic Prompt
**System:** `You are a helpful AI assistant.`  
**User:** `Generate unit tests for tf.keras.layers.Dense`

**Issues:**
- No proper test structure (missing unittest framework)
- No main function for execution
- Basic assertions only

### Version 2: Improved Structure
**System:** `You are a unit test generator for tensorflow library.`  
**User:** `Generate a python unit test case to test tf.keras.layers.Dense API. Use unittest library.`

**Improvements:**
- Added unittest framework structure
- Proper test class with inheritance
- Better assertions using unittest methods
- Main function for execution

**Remaining Issues:**
- Limited test scenarios
- Missing "maximum coverage" instruction
- Missing some necessary imports

### Version 3: Final Zero-Shot Baseline
**System:** `You are a unit test suite generator for {library} library.`  
**User:** `Generate a python unit test case to test the functionality of {api} API in {library} library with maximum coverage. Only create new tests if they cover new lines of code. Generate test suite using unittest library so it can be directly runnable (with the necessary imports and a main function).`

**Final Improvements:**
- Added "maximum coverage" focus
- Included "only create new tests if they cover new lines" constraint
- Added necessary imports
- More comprehensive test scenarios
- Emphasized "directly runnable" requirement

### Version 4: RAG-Enhanced
**System:** `You are a unit test suite generator for {library} library.`  
**User:** `Generate a python unit test case to test the functionality of {api} API in {library} library with maximum coverage. Use the following documents (surrounded by @@@) to make the test case more compilable, and passable, and cover more lines. Only create new tests if they cover new lines of code. Generate test suite using unittest library so it can be directly runnable (with the necessary imports and a main function).`

```
@@@ Doc_1: {retrieved_doc_1} @@@
@@@ Doc_2: {retrieved_doc_2} @@@
@@@ Doc_3: {retrieved_doc_3} @@@
```

**RAG Enhancements:**
- More comprehensive test coverage from retrieved documentation
- Emphasize on compilable, passable, and more coverage

## 📊 Key Improvements Through Evolution

The iterative refinement process led to notable improvements:

### Version 1 → Version 2
- **Structure**: Added proper unittest framework
- **Organization**: Better code organization with test classes
- **Execution**: Added main function for direct execution

### Version 2 → Version 3
- **Coverage Focus**: Added "maximum coverage" instruction
- **Completeness**: Emphasized "directly runnable" requirement
- **Line Coverage**: Added constraint to only test new lines
- **Comprehensiveness**: More test scenarios and error handling

### Version 3 → Version 4
- **Context Integration**: Added retrieved documentation guidance
- **Advanced Testing**: More sophisticated test patterns
- **Real-world Patterns**: Community-driven test scenarios
- **Quality**: Better assertions and validations

## 🧪 Example Files

- `version1.py`: Initial basic prompt output
- `version2.py`: Improved structure prompt output
- `version3.py`: Final zero-shot baseline prompt output (used in research)
- `version4.py`: RAG-enhanced prompt output (used in research)

## 🔍 Context Sources for RAG

For the RAG-enhanced approach, we retrieved relevant context from:

1. **API Documentation**: Official library documentation with usage examples
2. **GitHub Issues**: Community-reported bugs and edge cases
3. **StackOverflow**: Q&A discussions with practical usage patterns

## 💡 Design Lessons

1. **Iterative Refinement**: Each version addressed specific limitations
2. **Structure Matters**: Proper unittest framework is essential
3. **Coverage Focus**: Explicit coverage instructions improve test quality
4. **Context Integration**: Retrieved documentation significantly enhances results
5. **Practical Emphasis**: "Directly runnable" requirement improves code quality

This evolution demonstrates our prompt refinement to improve automated test generation from basic functionality to comprehensive, context-aware testing. The actuall prompts we used in code can be found in `api_rag.py`.

Note that these are few examples of the process taken, which we didn't make an extensive record at the time. However, it reflects similarly to as how we evolved our initial prompts.