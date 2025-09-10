import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import {
  LogOut,
  MessageCircle,
  MessageSquare,
  Moon,
  PlusCircle,
  Sun,
  Trash2,
  X,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Channel, ChannelFilters, ChannelSort } from "stream-chat";
import { ChannelList, useChatContext } from "stream-chat-react";
import { useTheme } from "../hooks/use-theme";

interface ChatSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onLogout: () => void;
  onNewChat: () => void;
  onChannelDelete: (channel: Channel) => void;
}

const ChannelListEmptyStateIndicator = () => (
  <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
    <div className="mb-4">
      <div className="w-16 h-16 bg-gradient-to-br from-primary/15 via-primary/8 to-transparent rounded-2xl flex items-center justify-center shadow-sm border border-primary/10">
        <MessageCircle className="h-8 w-8 text-primary/70" />
      </div>
    </div>
    <div className="space-y-2 max-w-xs">
      <h3 className="text-sm font-medium text-foreground">
        No writing sessions yet
      </h3>
      <p className="text-xs text-muted-foreground leading-relaxed">
        Start a new writing session to begin creating content with your AI
        assistant.
      </p>
    </div>
    <div className="mt-4 flex items-center gap-1 text-xs text-muted-foreground/60">
      <span>Click "New Writing Session" to get started</span>
    </div>
  </div>
);

export const ChatSidebar = ({
  isOpen,
  onClose,
  onLogout,
  onNewChat,
  onChannelDelete,
}: ChatSidebarProps) => {
  const { client, setActiveChannel } = useChatContext();
  const { user } = client;
  const { theme, setTheme } = useTheme();
  const navigate = useNavigate();

  if (!user) return null;

  const filters: ChannelFilters = {
    type: "messaging",
    members: { $in: [user.id] },
  };
  const sort: ChannelSort = { last_message_at: -1 };
  const options = { state: true, presence: true, limit: 10 };

  return (
    <>
      {/* Backdrop for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* The Sidebar */}
      <div
        className={cn(
          "fixed lg:static inset-y-0 left-0 z-50 w-80 bg-background border-r flex flex-col transform transition-transform duration-300 ease-in-out",
          isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        )}
      >
        {/* Header */}
        <div className="p-4 border-b flex justify-between items-center">
          <h2 className="text-lg font-semibold">Writing Sessions</h2>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="lg:hidden h-8 w-8"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Channel List */}
        <ScrollArea className="flex-1">
          <div className="p-4 space-y-0">
            <ChannelList
              filters={filters}
              sort={sort}
              options={options}
              EmptyStateIndicator={ChannelListEmptyStateIndicator}
              Preview={(previewProps) => (
                <div
                  className={cn(
                    "flex items-center p-2 rounded-lg cursor-pointer transition-colors relative group mb-1",
                    previewProps.active
                      ? "bg-primary/20 text-primary-foreground"
                      : "hover:bg-muted/50"
                  )}
                  onClick={() => {
                    setActiveChannel(previewProps.channel);
                    navigate(`/chat/${previewProps.channel.id}`);
                    onClose();
                  }}
                >
                  <MessageSquare className="h-4 w-4 mr-2" />
                  <span className="flex-1 truncate text-sm font-medium">
                    {previewProps.channel.data?.name || "New Writing Session"}
                  </span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute right-1 h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity z-10"
                    onClick={async (e) => {
                      e.stopPropagation();
                      onChannelDelete(previewProps.channel);
                    }}
                    title="Delete writing session"
                  >
                    <Trash2 className="h-4 w-4 text-muted-foreground/70 hover:text-destructive" />
                  </Button>
                </div>
              )}
            />
          </div>
        </ScrollArea>

        {/* New Chat Button */}
        <div className="p-2 border-t">
          <Button onClick={onNewChat} className="w-full justify-start">
            <PlusCircle className="mr-2 h-4 w-4" />
            New Writing Session
          </Button>
        </div>

        {/* User Profile / Logout */}
        <div className="p-2 border-t bg-background">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                className="w-full justify-start items-center p-2 h-auto"
              >
                <Avatar className="w-8 h-8 mr-2">
                  <AvatarImage src={user?.image} alt={user?.name} />
                  <AvatarFallback>
                    {user?.name?.charAt(0) || "U"}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1 text-left">
                  <p className="font-semibold text-sm truncate">{user?.name}</p>
                  <p className="text-xs text-muted-foreground">Online</p>
                </div>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-72" align="end">
              <DropdownMenuItem
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              >
                {theme === "dark" ? (
                  <Sun className="mr-2 h-4 w-4" />
                ) : (
                  <Moon className="mr-2 h-4 w-4" />
                )}
                <span>
                  Switch to {theme === "dark" ? "Light" : "Dark"} Theme
                </span>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={onLogout}>
                <LogOut className="mr-2 h-4 w-4" />
                <span>Log out</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </>
  );
};
