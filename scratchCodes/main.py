from core.helpers.graph import draw_graph
from core.memory.graph_with_basic_memory import GraphWithBasicMemory

agent = GraphWithBasicMemory()
agent.build()
draw_graph(agent.graph)