from hasher import Hasher

# Assuming you want to hash a list of passwords
passwords_to_hash = ["myPlainPassword123", "anotherSecret!"]
hasher = Hasher(passwords_to_hash)
hashed_passwords = hasher.generate()

# Now, hashed_passwords contains the bcrypt-hashed versions of your plain text passwords
print(hashed_passwords)