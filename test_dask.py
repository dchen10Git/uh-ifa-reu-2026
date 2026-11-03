from dask.distributed import Client
import rebound
import logging
logging.getLogger("distributed.comm.tcp").setLevel(logging.CRITICAL)

def final_a(e):
    sim = rebound.Simulation()
    sim.add(m=1.0)
    sim.add(m=1e-3, a=1.0, e=e)
    sim.add(m=1e-3, a=1.6)
    sim.integrate(1000.)
    return sim.particles[1].a

if __name__ == "__main__":
    client = Client()
    futures = client.map(final_a, [0.0, 0.05, 0.1, 0.15])
    print(client.gather(futures))
    client.close()


