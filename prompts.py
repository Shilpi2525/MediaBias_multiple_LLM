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
