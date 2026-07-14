try:
    from .capture_states import main
except ImportError:
    from capture_states import main


if __name__ == "__main__":
    main()
