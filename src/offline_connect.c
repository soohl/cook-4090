/* Loopback/Unix connect guard for trusted native inference libraries.
 * Loaded before Python and inherited by its subprocesses. UDP/raw Internet
 * sockets remain blocked by seccomp in src.offline.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>

int connect(int fd, const struct sockaddr *address, socklen_t length) {
    int allowed = 0;
    if (address && length >= sizeof(sa_family_t)) {
        if (address->sa_family == AF_UNIX) {
            allowed = 1;
        } else if (address->sa_family == AF_INET && length >= sizeof(struct sockaddr_in)) {
            const struct sockaddr_in *v4 = (const struct sockaddr_in *)address;
            allowed = (ntohl(v4->sin_addr.s_addr) >> 24) == 127;
        } else if (address->sa_family == AF_INET6 && length >= sizeof(struct sockaddr_in6)) {
            const struct sockaddr_in6 *v6 = (const struct sockaddr_in6 *)address;
            allowed = IN6_IS_ADDR_LOOPBACK(&v6->sin6_addr) ||
                (IN6_IS_ADDR_V4MAPPED(&v6->sin6_addr) && v6->sin6_addr.s6_addr[12] == 127);
        }
    }
    if (!allowed) {
        errno = EPERM;
        return -1;
    }
    int (*real_connect)(int, const struct sockaddr *, socklen_t) = dlsym(RTLD_NEXT, "connect");
    if (!real_connect) {
        errno = ENOSYS;
        return -1;
    }
    return real_connect(fd, address, length);
}
