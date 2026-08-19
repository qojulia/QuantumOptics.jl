.PHONY: comparative update-webpage

comparative:
	cd comparative && python3 runall.py

update-webpage: comparative
	cd comparative && julia make.jl
