import pickle


def read_track_embeddings():
    try:
        with open("track_embeddings.pkl", "rb") as f:
            track_embeddings = pickle.load(f)

        # Save to a text file
        with open("track_embeddings.txt", "w") as txt_file:
            for track_id, embedding in track_embeddings.items():
                txt_file.write(f"Track ID: {track_id}\n")
                txt_file.write(f"Embedding: {embedding}\n\n")

        print("Track embeddings have been saved to track_embeddings.txt")
    except FileNotFoundError:
        print("Error: track_embeddings.pkl not found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    read_track_embeddings()