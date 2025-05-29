SEARCH_QUERY = """

You are given a topic of interest.

{topic}

Extract the top articles which talk about this topic. Make sure these are the most recent articles.

"""


BIAS_QUERY = """
You are tasked with analyzing a series of documents to identify potential biases.

Here are the documents: {documents}

Please focus on identifying biases related to race, profession, gender, and value. If no biases are detected, categorize the document as "no-bias." 

For each document, provide the analysis in the format outlined below. If a link to the document is available, include it as well.

The output should follow this structure:

Document Number: This must be displayed as a small title with numbering order starting from 1.

Document Name: State the name or identifier of the document.

Link to the Document: Provide the hyperlink which is in href tag of the document or mention "Not available" if the link is not provided.

Bias Status: Indicate the type of bias detected, which could be one of the following: race, profession, gender, value, or no-bias.

Reason: Offer a concise explanation for why the document was categorized with a particular bias status, citing specific examples or language from the document.

Instructions:

1. Thoroughly review each document provided.

2. Pay attention to the language and content that might suggest biases.

3. Deliver a justification grounded in specific evidence from the document.

4. Keep the analysis clear, objective, and succinct.

"""



BIAS_DETECTION_PROMPT = """
You are an AI tasked with detecting **if a document contains any form of social bias**.

Below are the documents:

{documents}

Your goal is to determine whether each document is **biased** or **not biased**. Do not mention the type of bias — only assess the presence of bias.

---

### Output Format (for each document):

Document Number: (e.g., 1)  
Document Name: (Extract from document title or assign a simple name)  
Link to the Document: (Extract the URL from the <Document href="..."/> tag or say "Not available")  
Bias Status: (Choose one: biased / no-bias)  
Reason: (Briefly explain why the document was flagged as biased or no-bias, based on language, tone, or framing.)

---

### Guidelines:
- Focus on potentially loaded, prejudicial, or unbalanced language.
- If content appears neutral, label as "no-bias."
- Do **not** guess the type of bias (that will be handled by a second step).
"""





BIAS_TYPE_PROMPT = """
You are an expert in detecting specific types of bias.

Below is a bias detection result for documents that were already flagged as **biased**:

{biased_documents}

Your task is to identify **which type of bias** is present in each document. Use the following categories:
- Race
- Gender
- Profession
- Value

---

### Output Format:

Document Number: (same number as before)  
Bias Type: (One of: race, gender, profession, value)  
Reason: (Explain what aspect of the document supports this classification — refer to language or examples.)

---

If a document is not biased, skip it entirely (it was already filtered in the previous step).
"""
