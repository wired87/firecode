# coder

synthax: 
"::" next jump in local wf
"->" next pathway step
"-->" edge connect 

extend the following prompt with a clearer workflow definition and detailed tasks to improve the provided Code Module:
- create nx.MiltiDiGraph -> walk local dir (exclude .venv) -> extract content each file add_node file_name -> use ast: identify all present datatypes in each file -> extract and classify all content -> add_node for each with the parent=[parent file-node] and type=datatype(class, def, comment) ´, embed datatype (class/method etc) (only node keys wrapped inside dict(embedding, nid, t, type, code)
- collect all packages specified in r.txt -> add_node(dict(nid, package instance, embed description) 
- scan each datatype for sub modules/classes/functions/motheds/packages used from outer space -> link logical to destination(e.g. method A defines a class inside from external) -> add_edge with rel="uses" - classify directions (seppare between imports or imported)
  

# Step 2 (Query pipe):
**cli**:
@ cli start: create json file from created graph
- define workflow: function: receive string -> redirect to file search followign docs here: "https://ai.google.dev/gemini-api/docs/file-search?hl=de" with static pormpt and graph file. return: list[str] node_ids components 
  - include a pathfinding algirithmus whcih receivesa nid of a specific datatype e.g. class-> get_node(nid) -> get_neighbors rel="needs" -> collect neighbor nodes global in class attr "self.pathway_nodes"->           repeat process from AAA; output: all nods used by a specific datatype (like class) and allits sub modules  
  - for all params used(method/class header) by specific datatype: collect data from .env file (open...)
  - def: create and load entire str code base executable in ray.remote -> run function


extras:
- use clear oneliner comments before each fuction/method call and at the start of each method to intepret
- use creative emojicons
- include a r.txt (requiremens)
- follow functiona call schema from provided py content
- define the entire codebase inside a single file
- define a clear step by step workflow hat executes each possibel cli action wrapped isnide a testing def and if name amin call
- include funcitnoality to load picked datatypes(fucntions & classes) for all defined workflows inisde a ray.remote
- use pyvis for rendering the generated graph file
