"""
  This code is provided AS IS for example purpose and testing MD Device Server
  ARINAX Sep. 2021
"""

from time import sleep

#import epics
from GenericClient import GenericClient
import time

SIMULATION = False


class EpicsClientFactory:
    @staticmethod
    def instantiate(**kwargs):
        return EpicsClient(**kwargs)


class EpicsClient(GenericClient):

    def __init__(self, prefix):

        GenericClient.__init__(self)
        self.prefix = prefix + ":"
        self.hasCmdCache = {}       # a cache storing device server's methods than can be called with Epics
        self.hasAttrCache = {}      # a cache storing device server's attributes than can be read with Epics
        self.isEnum = {}            # a cache for attributes' and method returns' types (enum ==> read as str)
        self.timeout = 1 if SIMULATION else 5
        # self.__event = USE_EVENTS

    def __updateAttributeCache(self, attr_name, attr_is_avail, attr_is_enum):
        """
        This function updates the availability flag of an attribute in the attribute cache.
        It also indicates if the attribute is of type enum.
        :param str attr_name: name of the attribute which availability is updated
        :param bool attr_is_avail: a boolean to indicate if attribute is available in device server
        :param bool attr_is_enum: a boolean indicating if the attribute is of type enum
        :rtype: None
        """
        self.hasAttrCache[attr_name] = attr_is_avail
        self.isEnum[attr_name] = attr_is_enum
        return

    def __updateCommandCache(self, cmd_name, cmd_is_avail, cmd_is_enum):
        """
        This function updates the availability flag of a command in the command cache.
        It also indicates if the command's return is of type enum.
        :param str cmd_name: name of the command which availability is updated
        :param bool cmd_is_avail: a boolean to indicate if command is available in device server
        :param bool cmd_is_enum: a boolean indicating if the command's return is of type enum
        :rtype: None
        """
        self.hasCmdCache[cmd_name] = cmd_is_avail
        self.isEnum[cmd_name] = cmd_is_enum
        return

    def hasCmd(self, cmdName):
        """
        This function checks if a command in the device server is available from the Epics client.
        :param str cmdName: name of the command which availability is checked
        :rtype: bool
        """
        pv = self.prefix + cmdName
        pv = pv.replace(" ", "")
        if pv not in self.hasCmdCache.keys():
            info = epics.cainfo(pv, print_out=False, timeout=self.timeout)
            if info is not None:
                self.__updateCommandCache(pv, 'cannot connect' not in info, 'enum' in info)
        return self.hasCmdCache[pv]

    def hasAttribute(self, attrName):
        """
        This function checks if an attribute in the device server is available from the Epics client.
        :param str attrName: name of the attribute which availability is checked
        :rtype: bool
        """
        aName = self.prefix + attrName
        aName = aName.replace(" ", "")
        # print("aName ", aName)
        # print(type(aName))
        # print(epics.cainfo("MD3-Local:OmegaState"))
        # if (attrName.lower() == 'state'):
        #    return epics.caget(aName, as_string=True, timeout=self.timeout)
        if aName not in self.hasAttrCache.keys():
            info = epics.cainfo(aName, print_out=False, timeout=self.timeout)
            if info is not None:
                self.__updateAttributeCache(aName, 'cannot connect' not in info, 'enum' in info)
        return self.hasAttrCache[aName]

    def runCmd(self, cmdName, *args, **kwds):
        """
        This function is used to trigger a command in the MD server using the Epics protocol.
        :param str cmdName: name of the command to be run
        :param * args: list of arguments of the command
        :param ** kwds: a dictionary of keywords to use with the function (currently unused)
        :return: result of the command run by the MD app (formatted to string)
        :rtype: str
        """
        pvName = self.prefix+cmdName
        pvName = pvName.replace(" ", "")

        # If method does not take arguments, send keyword __EMPTY__
        if len(args) < 1:
            epics.caput(pvName, "__EMPTY__")
        # Else, format arguments to be sent
        else:
            newArgs=[]
            for x in args:
                if type(x) == float and str(x)[::-1].find(".") > 5:
                    x = "{:5f}".format(x)
                try:
                    newArgs.append("{:d}".format(x))
                except Exception:
                    newArgs.append(str(x))
            if len(newArgs) > 40:
                newArgs = newArgs[:40]
            epics.caput(pvName, " ".join(newArgs))
        res = epics.caget(pvName, use_monitor=False, timeout=self.timeout)
        if isinstance(res, str):
            if '\x1f' in res:
                res = res.split('\x1f')[1:]
        else:
            try:
                res = [x for x in res if x]
                if len(res) == 1:
                    res = res[0]
            except TypeError as te:
                # do nothing
                res = res
        return res

    def readAttribute(self, attrName):
        """
        This function is used to read an attribute provided by the MD device server through Epics protocol
        :param str attrName: name of the attribute which value is to be retrieved
        :return: value of the attribute formatted to string
        :rtype: str
        """
        aName = self.prefix + attrName
        aName = aName.replace(" ", "")

        if aName not in self.isEnum.keys():
            epics.ca.poll(evt=1.e-5, iot=0.1)
            info = epics.cainfo(aName, print_out=False, timeout=self.timeout)
            if info is not None:
                self.__updateAttributeCache(aName, 'cannot connect' not in info, 'enum' in info)

        res = epics.caget(aName, as_string=self.isEnum[aName], use_monitor=True, timeout=self.timeout)

        if isinstance(res, str) and '\x1f' in res:
            res = res.split('\x1f')[1:]
        # print("%s=%s" %(aName, res))
        return res

    def writeAttribute(self, attrName, value):
        """
        This function is used to write an attribute in the MD device server.
        :param str attrName: name of the attribute to be written
        :param int|str|float value: value to write in the attribute
        """
        if not isinstance(value, str):
            try:
                epics.caput(self.prefix + attrName, int(value[1]))
            except Exception as e:
                epics.caput(self.prefix + attrName, value)
        else:
            epics.caput(self.prefix+attrName, value)

    def onEpicsEvent(self, **kw):
        """
        This function is a callback for the Epics client subscriptions to channels of the device server.
        :param ** kw: dictionary of elements of the message
        :rtype: None
        """
        val = kw['value']
        if "enum" in kw['type']:
            val = kw['char_value']
        self.onEventReceived(kw['pvname'][len(self.prefix):], val, kw['timestamp']*1000)

    @staticmethod
    def nullWriter(txt):
        return

    def doSubscribe(self, attrName):
        """
        This function makes our client subscribe to an attribute's channel to get notified of its modifications
        :param str attrName: name of the monitored attribute
        :return: name of the attribute monitored\
        :rtype: str
        """
        epics.camonitor(self.prefix+attrName, self.nullWriter, self.onEpicsEvent)
        return attrName

    def doUnsubscribe(self, attrName, eventId):
        """
        This function unsubscribes the Epics client from events on an attribute.
        :param str attrName: name of the attribute to stop monitoring
        :param eventId:
        :rtype: None
        """
        epics.camonitor_clear(self.prefix+attrName)

    def getAttributesList(self):
        properties = self.getPropertiesSignature()
        print(properties)
        return None

    def hasEvents(self):
        #return self.__event
        return True
