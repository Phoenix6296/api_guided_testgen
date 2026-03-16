# Manual Analysis for RQ4 and RQ5

This directory contains the data, methodology, and results for our manual analysis conducted for Research Questions 4 and 5.

## 📋 Contents

### Data Files
- `reported_bugs.csv` - Bug reports discovered during testing (RQ4)
- `manual_data.csv` - Data analyzed during manual evaluation (RQ5)
- `detailed_analysis.csv` - Results and categorization from manual analysis (RQ5)

### Methodology Documentation
- `initial_categories.md` - Initial three-category framework
- `codebook.md` - Systematic taxonomy and coding framework
- `coding_process.md` - Independent coding and consensus building process

## 🔍 RQ4: Bug Reports

For RQ4, we analyzed and documented genuine bugs discovered during our testing across the ML libraries. The `reported_bugs.csv` file contains detailed information about each bug, including:

- Affected libraries and API
- Links to each GitHub issue report
- Each issue report has bug description, reproducing code, and expected/actual outcome
- Current status

## 🔬 RQ5: Manual Analysis Methodology

### Overview

For the manual analysis, we analyzed the retrieved documents, generated tests, and identified the unique code they covered. We documented which parts of the document might have led to the generated test that covered the unique code.

### Process Summary

We categorized documents based on their test generation impact using a systematic approach:

1. **Initial Categories**: Started with initial categories based on our knowledge of three document types: bug-inducing (issues), standard use cases (documentation), and unique/edge use cases (Q&As)

2. **Open-coding Thematic Analysis**: Performed open-coding thematic analysis and clustered codes through systematic taxonomy development method

3. **Code-book Development**: Built a concise code-book following established methodology

4. **Independent Coding**: Two participants independently applied the code-book across all documents

5. **Consensus Building**: Two additional participants were involved to resolve disagreements through consensus

### Final Categories

Through our systematic analysis, we identified the following document categories:

1. **Bug-inducing** - Documents that reveal actual software defects or incorrect behavior
2. **Standard use cases** - Documentation showing typical, expected API usage
3. **API usage sequences** - Documents demonstrating proper API call sequences and workflows
4. **Unique/complex input/parameters** - Examples with sophisticated or edge-case parameter usage
5. **Exception handling** - Documents showing proper error handling for expected conditions
6. **Integration patterns** - Examples of how APIs work together in larger systems
7. **Performance considerations** - Documents addressing performance optimization and best practices

### Quality Assurance

- **Dual coding**: Two participants independently performed the analysis
- **Consensus building**: Two additional participants resolved disagreements through discussion
- **Transparency**: Complete methodology and artifacts provided for reproducibility
- **Systematic validation**: Ensured consistency and reliability through structured process

## 📊 Key Insights

Our manual analysis revealed how different document types contribute to test generation quality:

- **Bug-inducing documents** help generate tests that catch edge cases and errors
- **Standard use cases** provide reliable patterns for basic functionality testing
- **API usage sequences** enable comprehensive workflow testing
- **Complex parameter examples** improve test coverage of advanced features
- **Exception handling patterns** enhance error testing capabilities
- **Integration patterns** support system-level test generation
- **Performance considerations** guide optimization-focused testing

## 🔄 Reproducibility

All artifacts from the manual analysis are provided to ensure:
- **Transparency**: Complete methodology documentation
- **Reproducibility**: All data and decisions are traceable
- **Verification**: Independent researchers can validate our findings
- **Extensibility**: Future work can build upon our categorization framework

---

This manual analysis contributes to understanding how different types of retrieved documents influence automated test generation quality and provides a foundation for future research in RAG-based testing approaches.