"""Cria artigo_cec/refs.bib = artigo/refs.bib (já verificadas) + referências evolutivas verificadas em 22/out/2026 (autores, venue e ano conferidos em fontes)."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "artigo" / "refs.bib").read_text(encoding="utf-8")
new = r"""
@article{salimans2017,
  author  = {Salimans, Tim and Ho, Jonathan and Chen, Xi and Sidor, Szymon and Sutskever, Ilya},
  title   = {Evolution Strategies as a Scalable Alternative to Reinforcement Learning},
  journal = {arXiv preprint arXiv:1703.03864},
  year    = {2017}
}

@article{such2017,
  author  = {Petroski Such, Felipe and Madhavan, Vashisht and Conti, Edoardo and Lehman, Joel and Stanley, Kenneth O. and Clune, Jeff},
  title   = {Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning},
  journal = {arXiv preprint arXiv:1712.06567},
  year    = {2017}
}

@article{bai2023,
  author  = {Bai, Hui and Cheng, Ran and Jin, Yaochu},
  title   = {Evolutionary Reinforcement Learning: A Survey},
  journal = {Intelligent Computing},
  volume  = {2},
  pages   = {0025},
  year    = {2023}
}

@article{hansen2001,
  author  = {Hansen, Nikolaus and Ostermeier, Andreas},
  title   = {Completely Derandomized Self-Adaptation in Evolution Strategies},
  journal = {Evolutionary Computation},
  volume  = {9},
  number  = {2},
  pages   = {159--195},
  year    = {2001}
}

@article{hansen2016,
  author  = {Hansen, Nikolaus},
  title   = {The {CMA} Evolution Strategy: A Tutorial},
  journal = {arXiv preprint arXiv:1604.00772},
  year    = {2016}
}

@inproceedings{perezliebana2013,
  author    = {Perez-Liebana, Diego and Samothrakis, Spyridon and Lucas, Simon M. and Rolfshagen, Philipp},
  title     = {Rolling Horizon Evolution versus Tree Search for Navigation in Single-Player Real-Time Games},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)},
  pages     = {351--358},
  year      = {2013}
}
"""
out = root / "artigo_cec"
out.mkdir(exist_ok=True)
(out / "refs.bib").write_text(base.rstrip("\n") + "\n" + new, encoding="utf-8", newline="\n")
print("ok", (base + new).count("@"))
