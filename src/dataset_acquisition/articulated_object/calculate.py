try:
    from .calc_states import main
except ImportError:
    from calc_states import main


if __name__ == "__main__":
    main()
