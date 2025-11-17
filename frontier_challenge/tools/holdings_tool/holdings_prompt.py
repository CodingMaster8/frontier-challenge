ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting company, bank, and asset names from financial queries.

Your task is to identify all company names, bank names, asset names, and ticker symbols mentioned in the user's query.

<extraction_guidelines>
1. Extract proper names of companies, banks, and financial institutions
2. Include both full names and commonly used abbreviations (e.g., "Petrobras" and "PETR4")
3. Recognize Brazilian companies and assets (Petrobras, Vale, Itaú, Bradesco, etc.)
4. Recognize international companies (Apple, Microsoft, Amazon, etc.)
5. Extract ticker symbols if mentioned (PETR4, VALE3, AAPL, etc.)
6. Include bond issuers (Brazilian government, US Treasury, etc.)
7. If the query is vague (e.g., "tech stocks"), return an empty list - we need specific names
8. Remove articles and prepositions (extract "Vale" from "in Vale")
9. Preserve proper capitalization for Brazilian companies
10. If multiple variations are mentioned, include all of them
</extraction_guidelines>

<examples>
Query: "Funds that invest in Petrobras"
Entities: ["Petrobras", "PETR4"]

Query: "Which funds hold Apple stock?"
Entities: ["Apple", "AAPL"]

Query: "Funds with Vale and Itaú holdings"
Entities: ["Vale", "VALE3", "Itaú", "Itau", "ITUB4"]

Query: "Exposure to Brazilian government bonds"
Entities: ["Brazilian government", "Brasil", "Tesouro Nacional"]

Query: "Funds investing in Microsoft and Amazon"
Entities: ["Microsoft", "MSFT", "Amazon", "AMZN"]

Query: "Show me tech funds"
Entities: []  # Too vague, no specific company names

Query: "Funds that own Bradesco"
Entities: ["Bradesco", "BBDC4"]
</examples>

Now extract entities from this query:

<query>{query}</query>

{format_instructions}
"""
