"""User interaction utilities."""
import sys
import queue
import threading


def ask_for_confirmation(timeout=60):
    """Ask the user for confirmation with a timeout, defaulting to yes."""
    q = queue.Queue()

    def get_input():
        try:
            sys.stderr.write(f"Save results and plots? (y/n) [auto-yes in {timeout}s]: ")
            sys.stderr.flush()
            q.put(sys.stdin.readline().strip().lower())
        except EOFError:
            q.put('y')
        except Exception as e:
            print(f"\nError reading input: {e}. Defaulting to yes.")
            q.put('y')

    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True
    input_thread.start()

    try:
        answer = q.get(timeout=timeout)
        if answer == 'n':
            print("\nUser chose not to save.")
            return False
        else:
            if answer != 'y':
                print(f"\nReceived '{answer}', interpreting as yes.")
            return True
    except queue.Empty:
        print(f"\nTimeout ({timeout}s) reached. Proceeding to save automatically.")
        return True
    finally:
        if input_thread.is_alive():
            pass
