# Multi-Domain Ontology & Bipartite Networking Strategy

## 1. Domain Diversity Context
This pipeline must dynamically adapt to three radically different data domains without cross-contaminating their schemas:
*   **Domain A: NSDAP (Historical):** Focuses on chronological radicalization, ideological affiliations, and interpersonal rivalries.
*   **Domain B: InfluenceWatch (Modern Political):** Focuses on "dark money" financial flows, shell company affiliations, board memberships, and lobbying networks.
*   **Domain C: Oregon OREM/OPAL (Logistical/Disaster Response):** Focuses on multi-agency emergency coordination (ODHS, OEM, NGOs), geographic jurisdictions, mass care capabilities, and disaster funding flows.

## 2. Dynamic Schema Routing
To prevent the LLM from trying to apply an "Emergency Response" relation to a 1930s historical figure, the pipeline must implement an **Ontology Factory Pattern**.
*   **Implementation Rule:** Before `langextract` is called, the pipeline must identify the domain of the source text. It will then dynamically load the appropriate Pydantic schema (e.g., `InfluenceWatchSchema`, `OremDisasterSchema`, or `NsdapSchema`) and inject only that specific list of valid relations into the Ollama prompt.

## 3. Bipartite and Tripartite Projections
In modern political networks and state disaster response, direct Person-to-Person connections are rare. Actors are connected via shared affiliations.
*   **Bipartite Structures:** The `NetworkX` module must natively support bipartite graph construction. If the text states "Person A and Person B sit on the board of PAC X," or "Agency A and NGO B both responded to the Almeda Fire," the LLM must extract the `Person -> Organization` ties, and `NetworkX` must project that into a unipartite `Person -> Person` graph based on the shared organizational node.
*   **Hyperedges for Disaster Response:** For the OREM/OPAL domain, relations often involve three or more entities (e.g., "Agency A distributed Grant B to Community C during Event D"). The extraction schema for this domain must support event-centric hyperedges (storing the event or grant as the central node that actors and agencies connect to).

## 4. Geospatial and Financial Edge Attributes
*   **Geospatial Grounding:** When parsing OREM/OPAL disaster response texts, edges must include an optional `jurisdiction` or `location` attribute (e.g., extracting that an emergency coordination applies specifically to "Klamath County" or "Tribal Lands"). 
*   **Financial Grounding:** When parsing InfluenceWatch texts, edges must include an optional `monetary_value` attribute to capture exact funding amounts between PACs and shell companies.