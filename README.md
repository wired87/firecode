# coder

synthax: 
"::" next jump in local wf
"->" next pathway step
"-->" edge connect 
"- receive code str ->" rcs

extend the following prompt with a clearer workflow definition and detailed tasks to improve the provided Code Module:

# Graph Engine
- - create nx.MiltiDiGraph
  - 
- walk local dir (exclude .venv) -> extract content each file add_node file_name -> use ast: identify all present datatypes in each file -> extract and classify all content -> add_node for each with the parent=[parent file-node] and type=datatype(class, def, comment) ´, embed datatype (class/method etc) (only node keys wrapped inside dict(embedding, nid, t, type, code)
- collect all packages specified in r.txt -> add_node(dict(nid, package instance, embed description) 
- scan each datatype for sub modules/classes/functions/motheds/packages used from outer space -> link logical to destination(e.g. method A defines a class inside from external) -> add_edge with rel="uses" - classify directions (seppare between imports or imported)
  

# Relay remote
- receive query string -> classify to options: run, adapt = keys of dict, create 5 versions of prompt using a local llm downloadable with pip > 


# Collector remote
receive list query
- loop all nodes of the graph embed nid local  perform similarity-serach(ss) ->
  save all in dict(nid:score)
- loop nids with sc > .9:  
  - include a pathfinding algirithmus whcih receivesa nid of a specific datatype e.g. class-> get_node(nid) -> get_neighbors rel="needs" -> collect neighbor nodes global in class attr "self.pathway_nodes"->           repeat process from AAA; output: all nods used by a specific datatype (like class) and allits sub modules  
  - for all params used(method/class header) by specific datatype: collect data from .env file (open...)
  - collect the sorted and runnable code structure incl all variables as str -> return
 
# Executor remote
- rcs -> runnable and executte it by collectig all apckages from code Graph

# editor
- rcs -> llm call gem cli py client: static prompt: perform change on files -> write changes to Graph

# creator
- rcs -> 


# Debugger remote
- while loop all files in dir tmp/ray/session_latest: embed content perform classification "err" or "clean" if "err": extract pid from file_name.split(-)-1.split(.)0 -> extract ray actor handle by pid -> get code from remote name (find in graph) -> llm call input: error from file, detailed debug isntructions formulated based on error, . required output: adapted python string -> switch code in graph, keep prev code in "cache":str attr -> delete err file content   

run the extend and inmproved prompt to ensure functionality

# extras:
- use clear oneliner comments before each fuction/method call and at the start of each method to intepret
- use creative emojicons
- include th eentire setp to init and run ray 
- include a r.txt (requiremens)
- follow functiona call schema from provided py content
- define the entire codebase inside a single file
- define a clear step by step workflow hat executes each possibel cli action wrapped isnide a testing def and if name amin call
- include funcitnoality to load picked datatypes(fucntions & classes) for all defined workflows inisde a ray.remote
- use pyvis for rendering the generated graph file
