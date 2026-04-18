using System.Runtime.InteropServices;

namespace Llmdot.Metal.Native;

internal enum MtlResourceOptions : ulong
{
    StorageModeShared = 0,
    StorageModeManaged = 1 << 8,
    StorageModePrivate = 2 << 8
}

internal unsafe static class ObjC
{
    private const string LibObjC = "/usr/lib/libobjc.A.dylib";

    [DllImport(LibObjC, EntryPoint = "objc_getClass")]
    private static extern nint GetClassImpl(byte* name);

    [DllImport(LibObjC, EntryPoint = "sel_registerName")]
    private static extern nint RegisterSelectorImpl(byte* name);

    [DllImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static extern nint MsgSend(nint receiver, nint selector);

    [DllImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static extern nint MsgSend(nint receiver, nint selector, nint arg1);

    [DllImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static extern nint MsgSend(nint receiver, nint selector, nint arg1, nint arg2);

    [DllImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static extern nint MsgSend(nint receiver, nint selector, nint arg1, nint arg2, nint arg3);

    [DllImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static extern nint MsgSend(nint receiver, nint selector, nint arg1, nint arg2, nint arg3, nint arg4);

    [DllImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static extern nint MsgSend(nint receiver, nint selector, ulong arg1, ulong arg2);

    public static unsafe nint GetClass(string name)
    {
        fixed (byte* ptr = System.Text.Encoding.UTF8.GetBytes(name + "\0"))
            return GetClassImpl(ptr);
    }

    public static unsafe nint Sel(string name)
    {
        fixed (byte* ptr = System.Text.Encoding.UTF8.GetBytes(name + "\0"))
            return RegisterSelectorImpl(ptr);
    }

    public static unsafe nint CreateNSString(string str)
    {
        var nsStringClass = GetClass("NSString");
        var selAlloc = Sel("alloc");
        var selInit = Sel("initWithUTF8String:");
        var instance = MsgSend(nsStringClass, selAlloc);
        fixed (byte* ptr = System.Text.Encoding.UTF8.GetBytes(str + "\0"))
        {
            return MsgSend(instance, selInit, (nint)ptr);
        }
    }

    public static unsafe string GetString(nint nsString)
    {
        if (nsString == 0) return string.Empty;
        var selUtf8 = Sel("UTF8String");
        var utf8Ptr = MsgSend(nsString, selUtf8);
        return Marshal.PtrToStringUTF8(utf8Ptr) ?? string.Empty;
    }
}