import { ReactNode, useCallback } from "react";
import { User } from "stream-chat";
import { Chat, useCreateChatClient } from "stream-chat-react";
import { LoadingScreen } from "../components/loading-screen";
import { useTheme } from "../hooks/use-theme";

interface ChatProviderProps {
  user: User;
  children: ReactNode;
}

const apiKey = import.meta.env.VITE_STREAM_API_KEY as string;
const backendUrl = import.meta.env.VITE_BACKEND_URL as string;

if (!apiKey) {
  throw new Error("Missing VITE_STREAM_API_KEY in .env file");
}

export const ChatProvider = ({ user, children }: ChatProviderProps) => {
  const { theme } = useTheme();

  /**
   * Token provider function that fetches authentication tokens from our backend.
   * This is called automatically by the Stream Chat client when:
   * - Initial connection is established
   * - Token expires and needs refresh
   * - Connection is re-established after network issues
   */
  const tokenProvider = useCallback(async () => {
    if (!user) {
      throw new Error("User not available");
    }

    try {
      const response = await fetch(`${backendUrl}/token`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ userId: user.id }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to fetch token: ${errorText}`);
      }

      const { token } = await response.json();
      return token;
    } catch (err) {
      console.error("Error fetching token:", err);
      throw err;
    }
  }, [user]);

  /**
   * Create the Stream Chat client with automatic token management.
   * This handles:
   * - Initial authentication
   * - WebSocket connection management
   * - Automatic token refresh
   * - Real-time event handling
   */
  const client = useCreateChatClient({
    apiKey,
    tokenOrProvider: tokenProvider,
    userData: user,
  });

  // Show loading screen while client is being initialized
  if (!client) {
    return <LoadingScreen />;
  }

  return (
    <Chat
      client={client}
      theme={
        theme === "dark" ? "str-chat__theme-dark" : "str-chat__theme-light"
      }
    >
      {children}
    </Chat>
  );
};
